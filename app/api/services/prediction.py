import os
import tempfile
import joblib
import numpy as np
import pandas as pd
import uuid
import torch
import xgboost as xgb

from app.api.core.config import settings
from app.api.core.logger import setup_logger
from app.api.services.monitoring import save_prediction_log
from app.api.core.aws import get_s3_client
from app.api.services.s3 import read_csv_from_s3, write_csv_to_s3

logger = setup_logger(__name__)

# Cache em memória para os modelos (Singleton)
_model_cache = {}

import io
from concurrent.futures import ThreadPoolExecutor

def _load_from_s3_to_memory(s3_key: str):
    """
    Baixa um arquivo do S3 direto para um buffer em memória.
    """
    s3_client = get_s3_client()
    bucket = settings.S3_BUCKET_NAME
    logger.info(f"Carregando em memória: s3://{bucket}/{s3_key}")
    try:
        response = s3_client.get_object(Bucket=bucket, Key=s3_key)
        return response['Body'].read()
    except Exception as e:
        logger.warning(f"Artefato {s3_key} não encontrado ou erro no S3: {e}")
        return None

def _smart_load(buffer: bytes):
    """
    Detecta formato e carrega o objeto a partir do buffer de bytes.
    """
    if buffer is None: return None
    
    # Check Magic Number (PK for Zip/PyTorch)
    if buffer.startswith(b"PK"):
        return torch.load(io.BytesIO(buffer), map_location="cpu", weights_only=False)
    
    try:
        return joblib.load(io.BytesIO(buffer))
    except Exception:
        # Fallback para XGBoost
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(buffer)
            tmp_path = tmp.name
        try:
            model = xgb.XGBRegressor()
            model.load_model(tmp_path)
            return model
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)

def get_model_and_params(symbol: str):
    cache_key = f"{symbol}_model" 
    
    if cache_key not in _model_cache:
        logger.info(f"🚀 Carregamento PARALELO de artefatos para {symbol}...")
        
        keys = {
            "pipe": "models/pipeline/pipeline.pkl",
            "model": f"models/champion/modelo_{symbol}.pkl",
            "scaler": f"models/scaler/scaler_{symbol}.pkl"
        }
        
        # Baixa os 3 arquivos ao mesmo tempo (Paralelismo)
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {name: executor.submit(_load_from_s3_to_memory, key) for name, key in keys.items()}
            buffers = {name: f.result() for name, f in futures.items()}
        
        # Carrega os objetos
        _model_cache[f"{symbol}_pipeline"] = _smart_load(buffers["pipe"])
        _model_cache[f"{symbol}_model"]    = _smart_load(buffers["model"])
        _model_cache[f"{symbol}_scaler"]   = _smart_load(buffers["scaler"])
        
        logger.info("✅ Todos os artefatos carregados em memória.")
            
    return _model_cache[f"{symbol}_pipeline"], _model_cache[f"{symbol}_model"], _model_cache[f"{symbol}_scaler"]

def _model_predict(model, X_full: pd.DataFrame, n_expected_features: int) -> float:
    n_input_cols = X_full.shape[1]
    
    if isinstance(model, torch.nn.Module):
        model.eval()
        seq_len = n_expected_features // n_input_cols if n_expected_features > n_input_cols else 1
        X_window = X_full.tail(seq_len).values.astype("float32")
        X_tensor = torch.tensor(X_window, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(X_tensor)
        return float(output.squeeze().item())
    else:
        if n_expected_features > n_input_cols:
            seq_len = n_expected_features // n_input_cols
            X_window = X_full.tail(seq_len).values.flatten().reshape(1, -1)
            return float(model.predict(X_window)[0])
        return float(model.predict(X_full.tail(1))[0])

def pipe_to_predict(symbol: str, df_history: pd.DataFrame, df_macro: pd.DataFrame) -> dict:
    pipeline, model, scaler = get_model_and_params(symbol)
    
    ultimo_preco = float(df_history['Close'].iloc[-1])
    data_ref = df_history.index[-1].strftime('%Y-%m-%d')

    pipeline.is_training = False 
    df_limpo = pipeline.transform((df_history, df_macro))
    
    colunas_treino = [
        'SP500_Return', 'VIX_Return', 'EURUSD_Return', 'Sentiment_Score', 'Log_Return', 
        'Target_Log_Return', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_5', 'Rolling_Std_14', 
        'Distancia_SMA_20', 'Distancia_SMA_50', 'Bollinger_Width', 'Volume_Shock', 
        'Month_Sin', 'Month_Cos', 'Day_Sin', 'Day_Cos', 'Volume_ROC_5', 'OBV_ROC_5', 
        'RSI_14', 'MACD_Line', 'MACD_Signal', 'MACD_Histogram', 'ATR_14', 'ATR_Pct'
    ]
    
    X_full = df_limpo.reindex(columns=colunas_treino, fill_value=0.0)
    X_full = X_full.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Aplica Scaler de forma eficiente
    if scaler is not None and X_full.shape[1] == getattr(scaler, "n_features_in_", 0):
        X_full.iloc[:, :] = scaler.transform(X_full.values)

    # Predição
    n_expected = getattr(model, "n_features_in_", X_full.shape[1])
    if hasattr(model, "num_feature"): n_expected = model.num_feature()
    if hasattr(model, "num_features"): n_expected = model.num_features
    
    X_features = X_full.drop(columns=['Target_Log_Return']) if 'Target_Log_Return' in X_full.columns else X_full
    log_return_previsto_scaled = _model_predict(model, X_features, n_expected)
    
    # Inverte escala
    if scaler is not None:
        target_idx = list(X_full.columns).index('Target_Log_Return')
        dummy = np.zeros((1, X_full.shape[1]))
        dummy[0, target_idx] = log_return_previsto_scaled
        log_return_previsto = scaler.inverse_transform(dummy)[0, target_idx]
    else:
        log_return_previsto = log_return_previsto_scaled
        
    preco_previsto = ultimo_preco * np.exp(log_return_previsto)
    variacao_pct = (np.exp(log_return_previsto) - 1) * 100
    
    resultado = {
        "symbol": symbol,
        "current_price": round(float(ultimo_preco), 2),
        "predicted_price_tomorrow": round(float(preco_previsto), 2),
        "variation_pct": round(float(variacao_pct), 4),
        "timestamp": data_ref
    }
    
    # Log de monitoramento (S3)
    save_prediction_log(symbol, X_full.tail(1), resultado)
    
    return resultado