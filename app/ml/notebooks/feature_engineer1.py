# app/ml/pipeline/custom_transformers.py
import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, is_training=True):
        self.is_training = is_training

    def fit(self, X, y=None):
        return self
    
    def transform(self, X) -> pd.DataFrame:
        print(f"[Pipeline] Executando Pipeline 100% Pandas (Treinamento: {self.is_training})...")
        
        if isinstance(X, tuple) or isinstance(X, list):
            df_history_cru, df_macro_cru = X
        else:
            raise ValueError("O Pipeline exige uma tupla com (df_history, df_macro).")

        # ==========================================
        # 1. TRATAMENTO DO HISTÓRICO DA AÇÃO
        # ==========================================
        df = df_history_cru.copy()
        if 'Date' not in df.columns:
            df = df.reset_index()
            
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df = df.sort_values('Date').reset_index(drop=True)

        # ==========================================
        # 2. TRATAMENTO E MERGE DOS DADOS MACRO
        # ==========================================
        macro_pd = df_macro_cru.copy()
        if isinstance(macro_pd.columns, pd.MultiIndex):
            macro_pd = macro_pd['Close']
            
        if 'Date' not in macro_pd.columns:
            macro_pd = macro_pd.reset_index()
            
        macro_pd['Date'] = pd.to_datetime(macro_pd['Date']).dt.tz_localize(None)
        
        col_map = {'^GSPC': 'SP500_Close', '^VIX': 'VIX_Close', 'EURUSD=X': 'EURUSD_Close'}
        macro_pd = macro_pd.rename(columns={k: v for k, v in col_map.items() if k in macro_pd.columns})
        
        macro_pd['SP500_Return'] = np.log(macro_pd['SP500_Close'] / macro_pd['SP500_Close'].shift(1)).fillna(0)
        macro_pd['VIX_Return'] = np.log(macro_pd['VIX_Close'] / macro_pd['VIX_Close'].shift(1)).fillna(0)
        macro_pd['EURUSD_Return'] = np.log(macro_pd['EURUSD_Close'] / macro_pd['EURUSD_Close'].shift(1)).fillna(0)
        
        macro_cols = macro_pd[['Date', 'SP500_Return', 'VIX_Return', 'EURUSD_Return']]
        
        df = pd.merge(df, macro_cols, on='Date', how='left')
        df[['SP500_Return', 'VIX_Return', 'EURUSD_Return']] = df[['SP500_Return', 'VIX_Return', 'EURUSD_Return']].fillna(0)

        # ==========================================
        # 3. CONTEXTO DE SENTIMENTO
        # ==========================================
        print("[Pipeline] Injetando Sentimento das Noticias (FinBERT)...")
        caminho_news = os.path.join(BASE_DIR, 'app', 'ml', 'pipeline', 'race_news.csv')
        
        if os.path.exists(caminho_news):
            df_news = pd.read_csv(caminho_news)
            df_news['Date'] = pd.to_datetime(df_news['Date']).dt.tz_localize(None)
            df = pd.merge(df, df_news, on='Date', how='left')
            df['Sentiment_Score'] = df['Sentiment_Score'].fillna(0.0)
        else:
            df['Sentiment_Score'] = 0.0

        # ==========================================
        # 4. BASE, LAGS E OBV 
        # ==========================================
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        
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
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(span=14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        df['RSI_14'] = df['RSI_14'].fillna(100) 

        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_Line'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD_Line'] - df['MACD_Signal']
        
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - df['prev_close']).abs()
        tr3 = (df['Low'] - df['prev_close']).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR_14'] = true_range.rolling(window=14).mean()
        df['ATR_Pct'] = df['ATR_14'] / df['Close']

        # ==========================================
        # 7. LIMPEZA E RETORNO
        # ==========================================
        cols_to_drop = [
            'prev_close', 'Month', 'DayOfWeek', 'SMA_20', 'SMA_50', 
            'Close_Std_20', 'Bollinger_Upper', 'Bollinger_Lower', 
            'Vol_SMA_10', 'OBV_Signal', 'Open', 'High', 'Low', 'Close', 
            'Adj Close', 'Volume', 'OBV', 'Dividends', 'Stock Splits'
        ]
        
        cols_existentes = [c for c in cols_to_drop if c in df.columns]
        df = df.drop(columns=cols_existentes)
        
        cols_features = [
            c for c in df.columns if c not in ["Target_Log_Return", "Date"]
        ]
        
        # Antes de dropar, vamos salvar a última linha caso precisemos dela para predição
        df_last_row_backup = df.tail(1).copy()

        df = df.dropna(subset=cols_features)

        if df.empty and not self.is_training:
            print("[Pipeline] AVISO: Todas as linhas foram removidas pelo dropna. Usando fallback da última linha para predição.")
            df = df_last_row_backup.fillna(0.0)

        if self.is_training:
            df = df.dropna(subset=['Target_Log_Return'])
            
        print(f"[Pipeline] Pipeline concluido! Matriz pronta. Shape: {df.shape}")
        return df