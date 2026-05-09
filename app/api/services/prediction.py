import os
import tempfile
import joblib
import numpy as np
import pandas as pd

from app.api.core.config import settings
from app.api.core.logger import setup_logger
from app.api.services.monitoring import save_prediction_log
from app.api.core.aws import get_s3_client

# Importação dos serviços de S3
from app.api.services.s3 import read_csv_from_s3, write_csv_to_s3

logger = setup_logger(__name__)

# Cache em memória para os modelos (Singleton)
_model_cache = {}

def _download_model_from_s3(s3_key: str, local_temp_path: str):
    s3_client = get_s3_client()
    bucket = settings.S3_BUCKET_NAME
    logger.info(f"☁️ Baixando artefato do S3: s3://{bucket}/{s3_key}")
    try:
        s3_client.download_file(bucket, s3_key, local_temp_path)
    except Exception as e:
        logger.error(f"Falha ao baixar artefato do S3: {e}")
        raise

def get_model_and_params(symbol: str):
    cache_key = f"{symbol}_model" 
    
    if cache_key not in _model_cache:
        logger.info(f"⏳ Inicializando ML para {symbol}...")
        try:
            # 1. Pipeline de features
            pipeline_key = "models/pipeline/pipeline.pkl"
            pipe_path = os.path.join(tempfile.gettempdir(), "pipeline.pkl")
            _download_model_from_s3(pipeline_key, pipe_path) 
            
            # 2. Baixa o Modelo
            model_key = "models/champion/modelo.pkl"
            model_path = os.path.join(tempfile.gettempdir(), "modelo.pkl")
            _download_model_from_s3(model_key, model_path)
            
            # 3. Tenta baixar o Scaler 
            scaler = None
            scaler_key = "models/scaler/scaler.pkl"
            scaler_path = os.path.join(tempfile.gettempdir(), "scaler.pkl")
            
            try:
                _download_model_from_s3(scaler_key, scaler_path)
                scaler = joblib.load(scaler_path)
                logger.info("📐 Scaler carregado e ativado.")
            except Exception:
                logger.info("⚡ Nenhum scaler no S3. Assumindo modelo baseado em árvore.")
            
            # Salva no cache da memória RAM da Lambda
            _model_cache[f"{symbol}_pipeline"] = joblib.load(pipe_path)
            _model_cache[f"{symbol}_model"] = joblib.load(model_path)
            _model_cache[f"{symbol}_scaler"] = scaler
            
            logger.info("✅ Artefatos carregados na memória com sucesso!")
            
        except Exception as e:
            logger.error(f"❌ Erro Crítico ao carregar modelos: {e}")
            raise RuntimeError("Falha ao inicializar modelos de ML.")
            
    return _model_cache[f"{symbol}_pipeline"], _model_cache[f"{symbol}_model"], _model_cache[f"{symbol}_scaler"]

def pipe_to_predict(symbol: str, df_history: pd.DataFrame, df_macro: pd.DataFrame) -> dict:
    """
    Recebe os dados crus, processa no pipeline e gera a predição final.
    """
    logger.info(f"Iniciando pipe de inferência para {symbol}...")
    
    pipeline, model, scaler = get_model_and_params(symbol)

    ultimo_preco = float(df_history['Close'].iloc[-1])
    data_ref = df_history.index[-1].strftime('%Y-%m-%d')

    logger.info("⚙️ Pipeline transformando dados crus...")
    pipeline.is_training = False 
    
    df_limpo = pipeline.transform(df_history)
    
    if df_limpo.empty: 
        raise ValueError("DataFrame vazio após passar pelo pipeline. Verifique tratamento de nulos.")
        
    # ==========================================
    # LÓGICA DA JANELA DE TEMPO (O SEGREDO DOS 648)
    # ==========================================
    # Removemos a Data para deixar só os números, igual no treinamento
    df_features = df_limpo.drop(columns=['Date'], errors='ignore')
    
    # Preenche Nulos (se houver) e converte para Matriz NumPy
    dataset_cru = np.nan_to_num(df_features.values)
    
    if scaler is not None:
        logger.info("📐 Aplicando o Scaler nas features de entrada...")
        dataset_scaled = scaler.transform(dataset_cru)
    else:
        dataset_scaled = dataset_cru

    num_features = dataset_scaled.shape[1]
    
    # Descobre automaticamente o tamanho da janela de tempo do modelo campeão
    if hasattr(model, 'n_features_in_'):
        seq_length = int(model.n_features_in_ / num_features)
    elif hasattr(model, 'n_features_'): # Versões antigas do LightGBM
        seq_length = int(model.n_features_ / num_features)
    else:
        seq_length = 24 # Fallback para Deep Learning (LSTM/GRU)
        
    if len(dataset_scaled) < seq_length:
        raise ValueError(f"Histórico insuficiente. O modelo requer {seq_length} dias, mas temos apenas {len(dataset_scaled)}.")

    # Extrai exatamente os últimos N dias necessários
    janela_temporal = dataset_scaled[-seq_length:] # Shape exato da base do modelo
    
   # Verifica se é um modelo PyTorch (Rede Neural) ou Árvore
    is_pytorch = hasattr(model, 'eval') and hasattr(model, 'forward')
    
    if not is_pytorch:
        # Achata a matriz! Transforma ex: (24 dias, 27 features) em (1, 648)
        X_pred_final = janela_temporal.reshape(1, -1)
        logger.info(f"🧠 Gerando previsão de Árvore. Shape de entrada: {X_pred_final.shape}")
        log_return_previsto_cru = model.predict(X_pred_final)[0]
    else:
        # Cria um tensor 3D para a Rede Neural (1, 24, 27)
        import torch
        X_pred_final = torch.tensor(janela_temporal, dtype=torch.float32).unsqueeze(0)
        logger.info(f"🧠 Gerando previsão Neural. Shape de entrada: {X_pred_final.shape}")
        
        model.eval()
        with torch.no_grad():
            log_return_previsto_cru = model(X_pred_final).numpy().flatten()[0]

    # ==========================================
    # A MÁGICA DA DESNORMALIZAÇÃO (O MOTIVO DOS 600)
    # ==========================================
    if scaler is not None:
        # 1. Descobre em qual coluna o Target estava na hora do treinamento
        target_idx = list(df_features.columns).index('Target_Log_Return')
        
        # 2. Cria uma linha de zeros com o formato exato que o scaler espera (ex: 27 colunas)
        dummy_row = np.zeros((1, num_features))
        
        # 3. Injeta a predição escalonada na coluna correta
        dummy_row[0, target_idx] = log_return_previsto_cru
        
        # 4. Faz o caminho reverso (Inverse Transform) e extrai o valor real
        log_return_real = scaler.inverse_transform(dummy_row)[0, target_idx]
        logger.info(f"🔄 Desescalonando: de {log_return_previsto_cru:.4f} para {log_return_real:.4f}")
    else:
        log_return_real = float(log_return_previsto_cru)

    # ==========================================
    # CÁLCULOS E REGISTRO
    # ==========================================
    preco_previsto = ultimo_preco * np.exp(log_return_real)
    variacao_pct = (np.exp(log_return_real) - 1) * 100
    
    resultado = {
        "symbol": symbol,
        "current_price": round(ultimo_preco, 2),
        "predicted_price_tomorrow": round(preco_previsto, 2),
        "variation_pct": round(variacao_pct, 4),
        "timestamp": data_ref
    }
    
    # Salva o log do Evidently usando apenas o último dia como representação no JSON
    X_log_df = pd.DataFrame(janela_temporal[-1:], columns=df_features.columns)
    save_prediction_log(symbol, X_log_df, resultado)
    
    # ==========================================
    # CARGA INCREMENTAL DOS DADOS TRANSFORMADOS NO S3
    # ==========================================
    from app.api.services.cleaning import historical_cleaning
    
    cleaned_s3_key = f"data/processed/{symbol}_historical_cleaned.csv"
    df_cleaned_antigo = read_csv_from_s3(settings.S3_BUCKET_NAME, cleaned_s3_key)
    
    if df_cleaned_antigo is None or df_cleaned_antigo.empty:
        logger.warning(f"Arquivo {cleaned_s3_key} não existe. Acionando limpeza histórica completa...")
        historical_cleaning(symbol, df_macro)
    else:
        logger.info("🔄 Atualizando base limpa incrementalmente no S3...")
        
        df_novo = df_limpo.copy()
        if 'Date' not in df_novo.columns and (df_novo.index.name == 'Date' or isinstance(df_novo.index, pd.DatetimeIndex)):
            df_novo = df_novo.reset_index()
            if 'Date' not in df_novo.columns and 'index' in df_novo.columns:
                df_novo = df_novo.rename(columns={'index': 'Date'})
                
        if 'Date' in df_cleaned_antigo.columns:
            df_cleaned_antigo['Date'] = pd.to_datetime(df_cleaned_antigo['Date'])
            
        if 'Date' in df_novo.columns:
            df_novo['Date'] = pd.to_datetime(df_novo['Date'])
            
        df_consolidado = pd.concat([df_cleaned_antigo, df_novo])
        
        if 'Date' in df_consolidado.columns:
            df_consolidado = df_consolidado.drop_duplicates(subset=['Date'], keep='last')
            df_consolidado = df_consolidado.sort_values('Date')
            df_consolidado['Date'] = df_consolidado['Date'].dt.strftime('%Y-%m-%d')
            
        write_csv_to_s3(settings.S3_BUCKET_NAME, cleaned_s3_key, df_consolidado)
    
    return resultado