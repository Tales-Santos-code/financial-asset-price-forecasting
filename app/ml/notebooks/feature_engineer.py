import sys
import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import tempfile

# ==========================================
# HACK DE DIRETÓRIO (CORRIGIDO)
# Sobe 4 níveis: feature_engineer.py -> notebooks -> ml -> app -> RAIZ DO PROJETO
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

# Importações dos seus serviços internos
from app.api.services.finance_service import FinanceService # noqa: E402
from app.api.core.config import settings # noqa: E402
from app.api.core.aws import get_s3_client # noqa: E402
from app.api.services.s3 import read_csv_from_s3 # noqa: E402

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, is_training=True, symbol="RACE"):
        self.is_training = is_training
        self.symbol = symbol # Recebe o ticker dinamicamente para buscar a notícia certa

    def fit(self, X, y=None):
        return self
    
    def macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Instanciando o serviço apenas localmente para que ele não seja salvo no .pkl
        finance_service = FinanceService()
        
        # ==========================================
        # 2. CONTEXTO MACROECONÔMICO (S&P 500, VIX e Câmbio)
        # ==========================================
        min_date = df['Date'].min()
        max_date = df['Date'].max()
        
        print("🌐 Buscando Clima de Mercado Global (S&P 500, VIX, EUR/USD)...")
        macro_pd = finance_service.get_macro_data(min_date=min_date, max_date=max_date)
        
        if isinstance(macro_pd.columns, pd.MultiIndex):
            macro_pd = macro_pd['Close']
            
        macro_pd = macro_pd.reset_index()
        macro_pd['Date'] = pd.to_datetime(macro_pd['Date']).dt.tz_localize(None)
        
        macro_pd = macro_pd.rename(columns={
            '^GSPC': 'SP500_Close',
            '^VIX': 'VIX_Close',
            'EURUSD=X': 'EURUSD_Close'
        })
        
        macro_pd['SP500_Return'] = np.log(macro_pd['SP500_Close'] / macro_pd['SP500_Close'].shift(1))
        macro_pd['VIX_Return'] = np.log(macro_pd['VIX_Close'] / macro_pd['VIX_Close'].shift(1))
        macro_pd['EURUSD_Return'] = np.log(macro_pd['EURUSD_Close'] / macro_pd['EURUSD_Close'].shift(1))
        
        # Seleciona apenas as colunas de interesse
        macro_cols = macro_pd[['Date', 'SP500_Return', 'VIX_Return', 'EURUSD_Return']]
        
        # Faz o merge e preenche nulos com 0
        df = pd.merge(df, macro_cols, on='Date', how='left')
        df[['SP500_Return', 'VIX_Return', 'EURUSD_Return']] = df[['SP500_Return', 'VIX_Return', 'EURUSD_Return']].fillna(0)
        
        return df

    def transform(self, X) -> pd.DataFrame:
        print(f"Executando Pipeline 100% Pandas (Treinamento: {self.is_training})...")
        
        # ==========================================
        # 1. PREPARAÇÃO DA BASE
        # ==========================================
        df = X.copy()
        if 'Date' not in df.columns:
            df = df.reset_index()
            
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            df = df.sort_values('Date').reset_index(drop=True)

        # ==========================================
        # 2. CHAMADA MACROECONÔMICA
        # ==========================================
        df = self.macro_features(df)

        # ==========================================
        # 3. CONTEXTO DE SENTIMENTO (Notícias LLM via S3)
        # ==========================================
        print(f"🧠 Injetando Sentimento das Notícias (S3) para {self.symbol}...")
        
        # Nome do arquivo que criamos lá no seu Race_news_service.py
        s3_news_key = f"data/features/{self.symbol}_news_sentiment.csv"
        
        try:
            # Baixa direto da memória da AWS para o Pandas
            df_news = read_csv_from_s3(settings.S3_BUCKET_NAME, s3_news_key)
            
            if df_news is not None and not df_news.empty:
                df_news['Date'] = pd.to_datetime(df_news['Date']).dt.tz_localize(None)
                
                df = pd.merge(df, df_news, on='Date', how='left')
                df['Sentiment_Score'] = df['Sentiment_Score'].fillna(0.0)
            else:
                print("⚠️ Arquivo de notícias vazio ou não encontrado no S3. Criando coluna neutra.")
                df['Sentiment_Score'] = 0.0
                
        except Exception as e:
            print(f"⚠️ Erro ao buscar notícias no S3: {e}. Criando coluna neutra de segurança.")
            df['Sentiment_Score'] = 0.0

        # ==========================================
        # 4. BASE, LAGS E OBV (Cálculos Iniciais)
        # ==========================================
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # OBV Signal
        df['OBV_Signal'] = 0
        df.loc[df['Close'] > df['Close'].shift(1), 'OBV_Signal'] = df['Volume']
        df.loc[df['Close'] < df['Close'].shift(1), 'OBV_Signal'] = -df['Volume']
        df['OBV'] = df['OBV_Signal'].cumsum()
        
        df['prev_close'] = df['Close'].shift(1)
        df['Target_Log_Return'] = df['Log_Return'].shift(-1)
        
        for lag in [1, 2, 3, 5]:
            df[f'Lag_{lag}'] = df['Log_Return'].shift(lag)
            
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['Close_Std_20'] = df['Close'].rolling(window=20).std()
        
        df['Rolling_Std_14'] = df['Log_Return'].rolling(window=14).std()
        df['Vol_SMA_10'] = df['Volume'].rolling(window=10).mean()
        
        df['Month'] = df['Date'].dt.month
        df['DayOfWeek'] = df['Date'].dt.dayofweek

        # ==========================================
        # 5. ROCs, CALENDÁRIO E OSCILADORES
        # ==========================================
        df['Distancia_SMA_20'] = (df['Close'] / df['SMA_20']) - 1
        df['Distancia_SMA_50'] = (df['Close'] / df['SMA_50']) - 1
        
        df['Bollinger_Upper'] = df['SMA_20'] + (2 * df['Close_Std_20'])
        df['Bollinger_Lower'] = df['SMA_20'] - (2 * df['Close_Std_20'])
        df['Bollinger_Width'] = (df['Bollinger_Upper'] - df['Bollinger_Lower']) / df['SMA_20']
        
        df['Volume_Shock'] = df['Volume'] / df['Vol_SMA_10']
        
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['Day_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 5)
        df['Day_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 5)
        
        df['Volume_ROC_5'] = (df['Volume'] / df['Volume'].shift(5)) - 1
        df['OBV_ROC_5'] = (df['OBV'] / df['OBV'].shift(5)) - 1

        # ==========================================
        # 6. RSI, MACD E ATR
        # ==========================================
        # RSI 14
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(span=14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        df['RSI_14'] = df['RSI_14'].fillna(100) # Onde loss for 0

        # MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_Line'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD_Line'] - df['MACD_Signal']
        
        # ATR
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - df['prev_close']).abs()
        tr3 = (df['Low'] - df['prev_close']).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR_14'] = true_range.rolling(window=14).mean()
        df['ATR_Pct'] = df['ATR_14'] / df['Close']

        # ==========================================
        # 7. LIMPEZA INTELIGENTE E SEGURANÇA
        # ==========================================
        # Drop das colunas temporárias que criamos pelo caminho
        cols_to_drop = [
            'prev_close', 'Month', 'DayOfWeek', 'SMA_20', 'SMA_50', 
            'Close_Std_20', 'Bollinger_Upper', 'Bollinger_Lower', 
            'Vol_SMA_10', 'OBV_Signal', 'Open', 'High', 'Low', 'Close', 
            'Adj Close', 'Volume', 'OBV', 'Dividends', 'Stock Splits'
        ]
        
        # Só dropa as que realmente existem no DataFrame
        cols_existentes = [c for c in cols_to_drop if c in df.columns]
        df = df.drop(columns=cols_existentes)
        
        # Drop nulos decorrentes das médias móveis (exceto Target e Date)
        cols_features = [c for c in df.columns if c not in ['Target_Log_Return', 'Date']]
        print("Valores nulos por coluna antes do dropna:")
        print(df.isnull().sum())
        df = df.dropna(subset=cols_features)
        
        if self.is_training:
            df = df.dropna(subset=['Target_Log_Return'])
            
        # Fix: replace infinity with 0 to prevent exorbitant values in predictions
        df = df.replace([np.inf, -np.inf], 0)
            
        print("Pipeline concluído! Matriz purificada (Pandas) pronta para ML.")
        return df
    
if __name__ == "__main__":
    from app.ml.notebooks.feature_engineer import FeatureEngineering
    ticker_alvo = "RACE"
    get_data = FinanceService(ticker=ticker_alvo)
    
    print(f"📥 Baixando histórico da {ticker_alvo}...")
    df_history = get_data.get_historical_data(full=True, use_checkpoint=False)

    # Inicia a classe já com o símbolo da ação
    pipeline_ferrari = FeatureEngineering(is_training=True, symbol=ticker_alvo)
    
    x_transformado = pipeline_ferrari.fit_transform(df_history)
    
    # ==========================================
    # UPLOAD DIRETO PARA O S3
    # ==========================================
    print("☁️ Salvando pipeline treinado diretamente no S3...")
    
    bucket = settings.S3_BUCKET_NAME
    s3_client = get_s3_client()
    
    # Salva num arquivo temporário seguro pelo Windows/Linux
    temp_pipe_path = os.path.join(tempfile.gettempdir(), f"pipeline_temp_{ticker_alvo}.pkl")
    joblib.dump(pipeline_ferrari, temp_pipe_path)
    
    # Sobe pro Data Lake no caminho que a API vai ler
    s3_key = "models/pipeline/pipeline.pkl"
    s3_client.upload_file(temp_pipe_path, bucket, s3_key)
    
    # Faxina: apaga da sua máquina local
    if os.path.exists(temp_pipe_path):
        os.remove(temp_pipe_path)
        
    print(f"📦 Pipeline pronto para uso e enviado com sucesso para: s3://{bucket}/{s3_key}")