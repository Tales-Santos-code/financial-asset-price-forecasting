
import os
import tempfile
import joblib
import numpy as np
import pandas as pd
import torch

from app.api.core.config import settings
from app.api.core.logger import setup_logger
from app.api.services.monitoring import save_prediction_log
from app.api.core.aws import get_s3_client
from app.api.services.s3 import read_csv_from_s3, write_csv_to_s3

logger = setup_logger(__name__)
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
            pipeline_key = "models/pipeline/pipeline.pkl"
            pipe_path = os.path.join(tempfile.gettempdir(), f"pipeline_{symbol}.pkl")
            _download_model_from_s3(pipeline_key, pipe_path) 
            
            model_key = f"models/champion/modelo_{symbol}.pkl"
            model_path = os.path.join(tempfile.gettempdir(), f"modelo_{symbol}.pkl")
            _download_model_from_s3(model_key, model_path)
            
            scaler = None
            scaler_key = f"models/scaler/scaler_{symbol}.pkl"
            scaler_path = os.path.join(tempfile.gettempdir(), f"scaler_{symbol}.pkl")
            
            try:
                _download_model_from_s3(scaler_key, scaler_path)
                scaler = joblib.load(scaler_path)
                logger.info("📐 Scaler carregado e ativado.")
            except Exception:
                logger.info("⚡ Nenhum scaler no S3. Assumindo modelo baseado em árvore.")
            
            _model_cache[f"{symbol}_pipeline"] = joblib.load(pipe_path)
            try:
                _model_cache[f"{symbol}_model"] = joblib.load(model_path)
            except Exception:
                logger.info("🔄 Tentando carregar o modelo via PyTorch...")
                _model_cache[f"{symbol}_model"] = torch.load(model_path, weights_only=False)
                
            _model_cache[f"{symbol}_scaler"] = scaler
            
            logger.info("✅ Artefatos carregados na memória com sucesso!")
            
        except Exception as e:
            logger.error(f"❌ Erro Crítico ao carregar modelos: {e}")
            raise RuntimeError("Falha ao inicializar modelos de ML.")
            
    return _model_cache[f"{symbol}_pipeline"], _model_cache[f"{symbol}_model"], _model_cache[f"{symbol}_scaler"]

def pipe_to_predict(symbol: str, df_history: pd.DataFrame, df_macro: pd.DataFrame) -> dict:
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
    # CORREÇÃO FATO 1: SELEÇÃO EXATA DE FEATURES
    # ==========================================
    # Remove APENAS a Data. O Target_Log_Return DEVE ficar na matriz de features, 
    # pois o train_worker usou ele (via fatiamento ':') no create_sequences.
    # Remove a Data e remove obrigatoriamente a coluna do Gabarito (que estará vazia hoje)
    df_features = df_limpo.drop(columns=['Date', 'Target_Log_Return'], errors='ignore')
    
    # Se o modelo sabe quais colunas ele precisa, puxamos APENAS elas
    if hasattr(model, 'feature_name_'):
        colunas_exatas = [c for c in model.feature_name_ if c in df_features.columns]
        if colunas_exatas:
            df_features = df_features[colunas_exatas]
    elif hasattr(model, 'feature_names_in_'):
        colunas_exatas = [c for c in model.feature_names_in_ if c in df_features.columns]
        if colunas_exatas:
            df_features = df_features[colunas_exatas]
        
    dataset_cru = np.nan_to_num(df_features.values)
    
    if scaler is not None:
        logger.info("📐 Aplicando o Scaler nas features de entrada...")
        dataset_scaled = scaler.transform(dataset_cru)
    else:
        dataset_scaled = dataset_cru

    num_features = dataset_scaled.shape[1]
    
    # Prevenção absoluta do seq_length = 0 
    if hasattr(model, 'n_features_in_') and num_features > 0:
        seq_length = max(1, int(model.n_features_in_ / num_features))
    elif hasattr(model, 'n_features_') and num_features > 0:
        seq_length = max(1, int(model.n_features_ / num_features))
    else:
        seq_length = 24 
        
    if len(dataset_scaled) < seq_length:
        raise ValueError(f"Histórico insuficiente. O modelo requer {seq_length} dias, mas temos apenas {len(dataset_scaled)}.")

    # Fatiamento seguro 
    janela_temporal = dataset_scaled[-seq_length:] 
    
    is_pytorch = hasattr(model, 'eval') and hasattr(model, 'forward')
    
    if not is_pytorch:
        X_pred_final = janela_temporal.reshape(1, -1)
        logger.info(f"🧠 Gerando previsão de Árvore. Shape de entrada: {X_pred_final.shape}")
        log_return_previsto_cru = model.predict(X_pred_final)[0]
    else:
        X_pred_final = torch.tensor(janela_temporal, dtype=torch.float32).unsqueeze(0)
        logger.info(f"🧠 Gerando previsão Neural. Shape de entrada: {X_pred_final.shape}")
        
        model.eval()
        with torch.no_grad():
            log_return_previsto_cru = model(X_pred_final).numpy().flatten()[0]

    # ==========================================
    # CORREÇÃO FATO 2: DESNORMALIZAÇÃO DO PREÇO
    # ==========================================
    if scaler is not None:
        try:
            target_idx = list(df_limpo.columns).index('Target_Log_Return') - 1
            dummy_row = np.zeros((1, scaler.n_features_in_))
            dummy_row[0, target_idx] = log_return_previsto_cru
            log_return_real = scaler.inverse_transform(dummy_row)[0, target_idx]
            logger.info(f"🔄 Desescalonando: de {log_return_previsto_cru:.4f} para {log_return_real:.4f}")
        except Exception:
            logger.warning("Falha na desnormalização exata. Usando valor cru.")
            log_return_real = float(log_return_previsto_cru)
    else:
        # ATENÇÃO: Se não há scaler no S3, o modelo DEVE ter sido treinado sem escalar o 'y'!
        log_return_real = float(log_return_previsto_cru)

    preco_previsto = ultimo_preco * np.exp(log_return_real)
    variacao_pct = (np.exp(log_return_real) - 1) * 100
    
    resultado = {
        "symbol": symbol,
        "current_price": round(ultimo_preco, 2),
        "predicted_price_tomorrow": round(preco_previsto, 2),
        "variation_pct": round(variacao_pct, 4),
        "timestamp": data_ref
    }
    
    X_log_df = pd.DataFrame(janela_temporal[-1:], columns=df_features.columns)
    save_prediction_log(symbol, X_log_df, resultado)
    
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