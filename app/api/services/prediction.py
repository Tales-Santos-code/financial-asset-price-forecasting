import joblib
import numpy as np
import pandas as pd


from app.api.core.config import settings
from app.api.core.logger import setup_logger
from app.api.services.monitoring import save_prediction_log

logger = setup_logger(__name__)

# Cache em memória para os modelos (Singleton)
_model_cache = {}

def get_model_and_params():
    """Carrega os artefatos apenas na primeira vez que a API for chamada."""
    if "pipeline" not in _model_cache:
        logger.info("⏳ Carregando artefatos de Machine Learning na memória...")
        try:
            _model_cache["pipeline"] = joblib.load(settings.PIPELINE_PATH)
            _model_cache["model"] = joblib.load(settings.MODEL_PATH)
            logger.info("✅ Pipeline e Modelo carregados com sucesso!")
        except Exception as e:
            logger.error(f"❌ Erro Crítico ao carregar modelos: {e}")
            raise RuntimeError("Falha ao inicializar modelos de ML.")
            
    return _model_cache["pipeline"], _model_cache["model"]

def pipe_to_predict(symbol: str, df_history: pd.DataFrame, df_macro: pd.DataFrame) -> dict:
    """
    Recebe os dados crus, processa no pipeline e gera a predição final.
    """
    logger.info(f"Iniciando pipe de inferência para {symbol}...")
    pipeline, model = get_model_and_params()

    # O yfinance geralmente traz a data no index
    ultimo_preco = float(df_history['Close'].iloc[-1])
    data_ref = df_history.index[-1].strftime('%Y-%m-%d')

    logger.info("⚙️ Pipeline limpando e mergeando arquivos crus...")
    pipeline.is_training = False 
    
    # Passamos a tupla rigorosamente como a sua classe espera
    df_limpo = pipeline.transform((df_history, df_macro))
    
    features_hoje = df_limpo.tail(1)
    if features_hoje.empty: 
        raise ValueError("DataFrame vazio após passar pelo pipeline. Verifique tratamento de nulos.")
        
    cols_para_remover = ['Target_Log_Return', 'Date']
    X_pred = features_hoje.drop(columns=[c for c in cols_para_remover if c in features_hoje.columns])
    
    logger.info("🧠 Gerando previsão com XGBoost...")
    log_return_previsto = model.predict(X_pred)[0]
    preco_previsto = ultimo_preco * np.exp(log_return_previsto)
    variacao_pct = (np.exp(log_return_previsto) - 1) * 100
    
    resultado = {
        "symbol": symbol,
        "current_price": round(ultimo_preco, 2),
        "predicted_price_tomorrow": round(preco_previsto, 2),
        "variation_pct": round(variacao_pct, 4),
        "timestamp": data_ref
    }
    
    
    # Passamos o X_pred (features) e o resultado para o S3
    save_prediction_log(symbol, X_pred, resultado)
    
    return resultado