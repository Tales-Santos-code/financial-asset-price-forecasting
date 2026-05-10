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
    logger.info(f"Baixando artefato do S3: s3://{bucket}/{s3_key}")
    try:
        s3_client.download_file(bucket, s3_key, local_temp_path)
    except Exception as e:
        logger.error(f"Falha ao baixar artefato do S3: {e}")
        raise

def _load_artifact(path: str):
    """
    Carrega um artefato detectando automaticamente o formato:
    - Comeca com b'PK' -> PyTorch (torch.save, formato zip)
    - Caso contrario  -> joblib.load
    """
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic == b"PK":
        logger.info(f"Detectado formato PyTorch: {path}")
        import torch
        return torch.load(path, map_location="cpu", weights_only=False)
    else:
        try:
            logger.info(f"Tentando formato joblib: {path}")
            return joblib.load(path)
        except Exception:
            logger.info(f"Falhou no joblib, tentando XGBoost (.ubj): {path}")
            import xgboost as xgb
            model = xgb.XGBRegressor()
            model.load_model(path)
            return model

def get_model_and_params(symbol: str):
    cache_key = f"{symbol}_model" 
    
    if cache_key not in _model_cache:
        logger.info(f"Inicializando ML para {symbol}...")
        try:
            import uuid
            unique_id = uuid.uuid4().hex
            
            # 1. Download Pipeline de Engenharia de Atributos
            pipe_key = "models/pipeline/pipeline.pkl"
            pipe_path = os.path.join(tempfile.gettempdir(), f"pipeline_{symbol}_{unique_id}.pkl")
            _download_model_from_s3(pipe_key, pipe_path)
            
            # 2. Download do Modelo Campeao Especifico da Acao
            model_key = f"models/champion/modelo_{symbol}.pkl"
            model_path = os.path.join(tempfile.gettempdir(), f"modelo_{symbol}_{unique_id}.pkl")
            _download_model_from_s3(model_key, model_path)
            
            # 3. Tenta baixar o Scaler (O Maestro apaga se não precisar)
            scaler = None
            scaler_key = f"models/scaler/scaler_{symbol}.pkl"
            scaler_path = os.path.join(tempfile.gettempdir(), f"scaler_{symbol}_{unique_id}.pkl")
            
            try:
                _download_model_from_s3(scaler_key, scaler_path)
                scaler = joblib.load(scaler_path)
                logger.info("Scaler carregado e ativado.")
            except Exception:
                logger.info("Nenhum scaler no S3. Assumindo modelo baseado em arvore.")
            
            # Salva no cache usando o loader inteligente
            _model_cache[f"{symbol}_pipeline"] = _load_artifact(pipe_path)
            _model_cache[f"{symbol}_model"]    = _load_artifact(model_path)
            _model_cache[f"{symbol}_scaler"]   = scaler
            
            logger.info("Artefatos carregados na memoria com sucesso!")
            
            # Limpeza dos temporários para AWS Lambda (Evita erro: No space left on device)
            for temp_file in [pipe_path, model_path, scaler_path]:
                try:
                    if temp_file and os.path.exists(temp_file):
                        os.remove(temp_file)
                except OSError as e:
                    logger.debug(f"Aviso de lock do SO no arquivo temporário: {e}")
            

        except Exception as e:
            logger.error(f"Erro Critico ao carregar modelos: {e}")
            raise RuntimeError("Falha ao inicializar modelos de ML.")
            
    return _model_cache[f"{symbol}_pipeline"], _model_cache[f"{symbol}_model"], _model_cache[f"{symbol}_scaler"]

def _model_predict(model, X_full: pd.DataFrame, n_expected_features: int) -> float:
    """
    Executa inferencia detectando se o modelo espera uma linha simples (27),
    uma janela achatada (ex: 24*27=648) ou um tensor 3D (LSTM).
    """
    import torch
    n_input_cols = X_full.shape[1]
    
    # Caso 1: Modelo PyTorch (LSTM/GRU)
    if isinstance(model, torch.nn.Module):
        logger.info("Usando inferencia PyTorch (nn.Module).")
        model.eval()
        
        # Se o modelo espera ex: 648 features, precisamos de 24 linhas de history
        seq_len = n_expected_features // n_input_cols if n_expected_features > n_input_cols else 1
        X_window = X_full.tail(seq_len).values.astype("float32")
        
        # LSTM espera (batch, seq_len, features)
        X_tensor = torch.tensor(X_window, dtype=torch.float32).unsqueeze(0) # (1, seq, feat)
        
        with torch.no_grad():
            output = model(X_tensor)
        return float(output.squeeze().item())

    # Caso 2: Modelos Sklearn/LGBM/XGBoost
    else:
        logger.info("Usando inferencia sklearn (.predict).")
        
        # Se o modelo espera ex: 648 features, ele quer a janela achatada (flattened)
        if n_expected_features > n_input_cols:
            seq_len = n_expected_features // n_input_cols
            logger.info(f"Modelo espera janela achatada: {seq_len} passos de tempo.")
            X_window = X_full.tail(seq_len).values.flatten().reshape(1, -1)
            return float(model.predict(X_window)[0])
        else:
            # Caso simples (1 linha)
            return float(model.predict(X_full.tail(1))[0])

def pipe_to_predict(symbol: str, df_history: pd.DataFrame, df_macro: pd.DataFrame) -> dict:
    """
    Recebe os dados crus, processa no pipeline e gera a predição final.
    """
    logger.info(f"Iniciando pipe de inferência para {symbol}...")
    
    # IMPORTANTE: Desempacota o scaler e passa o symbol para a função!
    pipeline, model, scaler = get_model_and_params(symbol)

    ultimo_preco = float(df_history['Close'].iloc[-1])
    data_ref = df_history.index[-1].strftime('%Y-%m-%d')

    logger.info("Pipeline transformando dados crus...")
    pipeline.is_training = False 
    
    df_limpo = pipeline.transform((df_history, df_macro))
    
    # Lista de colunas EXATA usada no treinamento (incluindo o Target bugado)
    # se a ordem mudar, o modelo erra a predição silenciosamente!
    colunas_treino = [
        'SP500_Return', 'VIX_Return', 'EURUSD_Return', 'Sentiment_Score', 'Log_Return', 
        'Target_Log_Return', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_5', 'Rolling_Std_14', 
        'Distancia_SMA_20', 'Distancia_SMA_50', 'Bollinger_Width', 'Volume_Shock', 
        'Month_Sin', 'Month_Cos', 'Day_Sin', 'Day_Cos', 'Volume_ROC_5', 'OBV_ROC_5', 
        'RSI_14', 'MACD_Line', 'MACD_Signal', 'MACD_Histogram', 'ATR_14', 'ATR_Pct'
    ]
    
    # Reordena e garante que todas existam (preenche Target com 0 se faltar)
    X_full = df_limpo.reindex(columns=colunas_treino, fill_value=0.0)
    X_full = X_full.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    logger.info(f"Features organizadas ({X_full.shape[1]}) para o modelo.")
    
    if X_full.empty: 
        raise ValueError("DataFrame vazio após passar pelo pipeline. Verifique tratamento de nulos.")

    # ==========================================
    # LÓGICA DE ESCALONAMENTO CONDICIONAL
    # ==========================================
    X_full_scaled = X_full.copy()
    if scaler is not None:
        n_expected = getattr(scaler, "n_features_in_", X_full.shape[1])
        n_atual = X_full.shape[1]
        
        if n_atual == n_expected:
            logger.info("Aplicando o Scaler em todo o historico transformado...")
            X_full_scaled = pd.DataFrame(
                scaler.transform(X_full.values),
                columns=X_full.columns,
                index=X_full.index
            )
        else:
            logger.warning(f"Mismatch no Scaler: pipeline {n_atual}, scaler {n_expected}. Ignorando.")

    # Descobre quantas features o modelo espera no total (ex: 648 para LGBM com 24 lags)
    n_expected_features = X_full.shape[1] # Default: 1 linha
    
    import torch
    if isinstance(model, torch.nn.Module):
        # PyTorch: Tentamos inferir pela entrada (ex: se for LSTM, costumamos salvar input_size)
        n_expected_features = getattr(model, "input_size", X_full.shape[1])
    else:
        # Sklearn/LGBM/XGBoost
        if hasattr(model, "n_features_in_"):
            n_expected_features = model.n_features_in_
        elif hasattr(model, "num_feature"): # LightGBM Booster
            n_expected_features = model.num_feature()
        elif hasattr(model, "num_features"): # XGBoost Booster
            n_expected_features = model.num_features
        elif hasattr(model, "feature_names_in_"):
            n_expected_features = len(model.feature_names_in_)

    logger.info(f"Gerando previsao. Modelo espera total de {n_expected_features} features.")
    
    # IMPORTANTE: O modelo foi treinado SEM a coluna alvo (Target_Log_Return) nas features
    # Portanto, precisamos removê-la da matriz antes de enviar para predição
    if 'Target_Log_Return' in X_full_scaled.columns:
        X_features_scaled = X_full_scaled.drop(columns=['Target_Log_Return'])
    else:
        X_features_scaled = X_full_scaled
        
    log_return_previsto_scaled = _model_predict(model, X_features_scaled, n_expected_features)
    
    # Inverte a escala do target para encontrar o retorno logarítmico real
    if scaler is not None:
        target_col_index = list(X_full.columns).index('Target_Log_Return')
        num_features = X_full.shape[1]
        dummy_preds = np.zeros((1, num_features))
        dummy_preds[0, target_col_index] = log_return_previsto_scaled
        log_return_previsto = scaler.inverse_transform(dummy_preds)[0, target_col_index]
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
    
    # Monitoramento (logamos apenas a ultima linha das features para analise de drift)
    save_prediction_log(symbol, X_full_scaled.tail(1), resultado)
    
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
        logger.info("Atualizando base limpa incrementalmente no S3...")
        
        # Garante que a coluna 'Date' exista como coluna real, não como index
        df_novo = df_limpo.copy()
        if 'Date' not in df_novo.columns and (df_novo.index.name == 'Date' or isinstance(df_novo.index, pd.DatetimeIndex)):
            df_novo = df_novo.reset_index()
            # Se o index não tinha nome, pode virar 'index'. Renomeamos para 'Date'
            if 'Date' not in df_novo.columns and 'index' in df_novo.columns:
                df_novo = df_novo.rename(columns={'index': 'Date'})
                
        # Converte a coluna Date de ambas as tabelas para Datetime (garante compatibilidade)
        if 'Date' in df_cleaned_antigo.columns:
            df_cleaned_antigo['Date'] = pd.to_datetime(df_cleaned_antigo['Date'])
            
        if 'Date' in df_novo.columns:
            df_novo['Date'] = pd.to_datetime(df_novo['Date'])
            
        # Junta o passado com as predições geradas hoje
        df_consolidado = pd.concat([df_cleaned_antigo, df_novo])
        
        # Remove duplicatas e organiza cronologicamente sem dar erro
        if 'Date' in df_consolidado.columns:
            # Mantém a última versão processada (a mais fresca)
            df_consolidado = df_consolidado.drop_duplicates(subset=['Date'], keep='last')
            df_consolidado = df_consolidado.sort_values('Date')
            
            # (Opcional, mas boa prática) Converte de volta para string YYYY-MM-DD para o CSV ficar limpo
            df_consolidado['Date'] = df_consolidado['Date'].dt.strftime('%Y-%m-%d')
            
        write_csv_to_s3(settings.S3_BUCKET_NAME, cleaned_s3_key, df_consolidado)
    
    return resultado