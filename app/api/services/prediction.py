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
            # 1. Pipeline de features (Baixa sempre o mesmo)
            pipe_path = os.path.join(tempfile.gettempdir(), f"pipeline_{symbol}.pkl")
            # Ajuste o nome "pipeline.pkl" se no seu S3 ele estiver dentro de alguma pasta (ex: "models/pipeline.pkl")
            _download_model_from_s3("pipeline.pkl", pipe_path) 
            
            # 2. Baixa o Modelo (O nome agora é sempre fixo graças ao Maestro)
            model_key = f"modelo_{symbol}.pkl"
            model_path = os.path.join(tempfile.gettempdir(), model_key)
            _download_model_from_s3(model_key, model_path)
            
            # 3. Tenta baixar o Scaler (O Maestro apaga se não precisar)
            scaler = None
            scaler_key = f"scaler_{symbol}.pkl"
            scaler_path = os.path.join(tempfile.gettempdir(), scaler_key)
            
            try:
                _download_model_from_s3(scaler_key, scaler_path)
                scaler = joblib.load(scaler_path)
                logger.info("📐 Scaler carregado e ativado.")
            except Exception:
                # Se cair aqui, é porque o arquivo não existe no S3 (modelo em árvore)
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
    
    # IMPORTANTE: Desempacota o scaler e passa o symbol para a função!
    pipeline, model, scaler = get_model_and_params(symbol)

    ultimo_preco = float(df_history['Close'].iloc[-1])
    data_ref = df_history.index[-1].strftime('%Y-%m-%d')

    logger.info("⚙️ Pipeline transformando dados crus...")
    pipeline.is_training = False 
    
    df_limpo = pipeline.transform((df_history, df_macro))
    
    features_hoje = df_limpo.tail(1)
    if features_hoje.empty: 
        raise ValueError("DataFrame vazio após passar pelo pipeline. Verifique tratamento de nulos.")
        
    cols_para_remover = ['Target_Log_Return', 'Date']
    X_pred = features_hoje.drop(columns=[c for c in cols_para_remover if c in features_hoje.columns])
    
    # ==========================================
    # LÓGICA DE ESCALONAMENTO CONDICIONAL
    # ==========================================
    X_pred_scaled = X_pred.copy()
    if scaler is not None:
        logger.info("📐 Aplicando o Scaler nas features de entrada...")
        # Aplica a transformação mantendo a estrutura do DataFrame do Pandas
        X_pred_scaled = pd.DataFrame(
            scaler.transform(X_pred), 
            columns=X_pred.columns, 
            index=X_pred.index
        )
    
    logger.info("🧠 Gerando previsão com o modelo campeão...")
    # Usa a versão escalonada (que pode estar transformada ou intacta dependendo do If acima)
    log_return_previsto = model.predict(X_pred_scaled)[0]
    preco_previsto = ultimo_preco * np.exp(log_return_previsto)
    variacao_pct = (np.exp(log_return_previsto) - 1) * 100
    
    resultado = {
        "symbol": symbol,
        "current_price": round(ultimo_preco, 2),
        "predicted_price_tomorrow": round(preco_previsto, 2),
        "variation_pct": round(variacao_pct, 4),
        "timestamp": data_ref
    }
    
    # Passamos o X_pred (features originais ou escaladas) e o resultado para o monitoramento
    save_prediction_log(symbol, X_pred_scaled, resultado)
    
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
        
        # Garante que a coluna 'Date' exista como coluna real, não como index
        df_novo = df_limpo.copy()
        if 'Date' not in df_novo.columns and (df_novo.index.name == 'Date' or isinstance(df_novo.index, pd.DatetimeIndex)):
            df_novo = df_novo.reset_index()
            # Se o index não tinha nome, pode virar 'index'. Renomeamos para 'Date'
            if 'Date' not in df_novo.columns and 'index' in df_novo.columns:
                df_novo = df_novo.rename(columns={'index': 'Date'})
                
        # --- CORREÇÃO DE TIPAGEM AQUI 👇 ---
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