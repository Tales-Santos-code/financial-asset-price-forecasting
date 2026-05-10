import pandas as pd
from app.api.core.config import settings
from app.api.core.logger import setup_logger
from app.api.services.s3 import read_csv_from_s3, write_csv_to_s3

# Importamos o carregador central para aproveitar o cache da RAM e o download do S3
from app.api.services.prediction import get_model_and_params

logger = setup_logger("cleaning_service")

def historical_cleaning(ticker: str, df_macro: pd.DataFrame = None):    
    logger.info(f"🔄 Iniciando limpeza histórica (Full Load) para {ticker}...")
    
    # 1. Lê o histórico master cru da nuvem
    df_history = read_csv_from_s3(bucket=settings.S3_BUCKET_NAME, key=f"data/historical/{ticker}_master_history.csv")
    
    if df_history is None or df_history.empty:
        logger.error(f"Master history não encontrado no S3 para {ticker}.")
        return
        
    # 2. Pega o pipeline já carregado na memória
    pipeline, _, _ = get_model_and_params(ticker)
    pipeline.is_training = False 
    
    # Como o seu pipeline aceita a tupla (history, macro), tentamos passar os dois.
    try:
        if df_macro is not None and not df_macro.empty:
            df_transformado = pipeline.transform((df_history, df_macro))
        else:
            df_transformado = pipeline.transform(df_history)
    except Exception as e:
        logger.warning(f"Fallback de transformação: {e}. Tentando sem tupla...")
        df_transformado = pipeline.transform(df_history)
    
    # 4. Salva de volta no S3
    caminho_s3 = f"data/processed/{ticker}_historical_cleaned.csv"
    write_csv_to_s3(bucket=settings.S3_BUCKET_NAME, key=caminho_s3, df=df_transformado)
    logger.info(f"📊 Base histórica limpa e salva com sucesso em: {caminho_s3}")