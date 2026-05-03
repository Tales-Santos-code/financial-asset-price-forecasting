import os
import tempfile
import pandas as pd

# ==========================================
# IMPORTAÇÕES DO EVIDENTLY (v0.4+) E S3
# ==========================================
from evidently import Report
from evidently.presets import DataDriftPreset

from app.api.core.config import settings
from app.api.core.logger import setup_logger

# Importamos todos os conectores necessários do S3
from app.api.services.s3 import write_html_to_s3, read_json_from_s3, read_csv_from_s3
from app.api.core.aws import get_s3_client

logger = setup_logger("drift_detector")

def load_production_logs(symbol: str) -> pd.DataFrame:
    """
    Lê os JSONs gerados pela API de predição DIRETAMENTE DO S3, sem tocar no disco.
    """
    s3_client = get_s3_client()
    bucket = settings.S3_BUCKET_NAME
    prefix = f"predictions/{symbol}/"
    
    records = []
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        
        if 'Contents' not in response:
            logger.info("Nenhum log de predição encontrado no S3 para análise.")
            return pd.DataFrame()
            
        for obj in response['Contents']:
            if obj['Key'].endswith('.json'):
                data = read_json_from_s3(bucket, obj['Key'])
                if data and "features_input" in data:
                    records.append(data["features_input"])
                    
    except Exception as e:
        logger.warning(f"Erro ao buscar logs de produção no S3: {e}")
        
    return pd.DataFrame(records)


def check_data_drift(symbol: str = "RACE") -> bool:
    """
    Execução Cloud Native (compatível com AWS Lambda e Evidently v0.4+).
    """
    logger.info(f"🔍 Iniciando análise de Data Drift para {symbol} (Cloud Native)...")
    
    bucket = settings.S3_BUCKET_NAME
    ref_s3_key = f"data/processed/{symbol}_historical_cleaned.csv"
    
    # 1. Busca os dados de referência (treino)
    try:
        reference_data = read_csv_from_s3(bucket, ref_s3_key)
        if reference_data is None or reference_data.empty:
            logger.error(f"Arquivo de referência não encontrado no S3: s3://{bucket}/{ref_s3_key}")
            return False
    except Exception as e:
        logger.error(f"Erro fatal ao ler referência do S3: {e}")
        return False
        
    # 2. Busca os dados de produção (inferência)
    current_data = load_production_logs(symbol) 
    
    if current_data.empty or len(current_data) < 5:
        logger.info("⚠️ Sem volume de dados suficiente em produção para analisar Drift estatístico.")
        return False

    features_para_analisar = [c for c in reference_data.columns if c not in ['Target_Log_Return', 'Date']]
    
    for col in features_para_analisar:
        if col not in current_data.columns:
            # Preenchemos com a média da referência para evitar o 'RuntimeWarning' de divisão por zero do NumPy
            current_data[col] = reference_data[col].mean() if not reference_data[col].empty else 0.0

    # ==========================================
    # ARQUITETURA IN-MEMORY PARA AWS (EVIDENTLY V0.7+)
    # ==========================================
    drift_report = Report([DataDriftPreset()])
    
    logger.info("⚙️ Calculando distribuições estatísticas nas variáveis...")
    
    # 1. O .run() devolve um objeto SNAPSHOT (Nova arquitetura do Evidently 0.7+)
    resultado_eval = drift_report.run(
        reference_data=reference_data[features_para_analisar], 
        current_data=current_data[features_para_analisar]
    )

    temp_html_path = os.path.join(tempfile.gettempdir(), f"drift_report_{symbol}.html")
    
    # 2. Chamamos os métodos de exportação direto no Snapshot
    resultado_eval.save_html(temp_html_path)
    
    # 3. Lê o HTML como string e injeta direto no S3
    with open(temp_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
        
    s3_key = f"monitoring/drift_reports/drift_report_{symbol}.html"
    write_html_to_s3(settings.S3_BUCKET_NAME, s3_key, html_content)
    
    logger.info(f"📊 Dashboard injetado direto no Data Lake: s3://{settings.S3_BUCKET_NAME}/{s3_key}")
    
    # 4. Limpa a memória temporária (Boa prática Serverless)
    if os.path.exists(temp_html_path):
        os.remove(temp_html_path)
    
    # 5. Extrai os metadados (JSON) com função recursiva blindada
    try:
        # Extrai o dicionário bruto
        report_json = resultado_eval.dict()
        
        # Deep Search: Acha a chave em qualquer nível do JSON
        def achar_drift(dicionario):
            if isinstance(dicionario, dict):
                if "dataset_drift" in dicionario:
                    return dicionario["dataset_drift"]
                for valor in dicionario.values():
                    res = achar_drift(valor)
                    if res is not None:
                        return res
            elif isinstance(dicionario, list):
                for item in dicionario:
                    res = achar_drift(item)
                    if res is not None:
                        return res
            return None

        dataset_drift = achar_drift(report_json)

        logger.info(f"Status do Data Drift extraído: {dataset_drift}")
        
        if dataset_drift is True:
            logger.warning("🚨 ALERTA VERMELHO: Data Drift Detectado! O comportamento do mercado mudou.")
            return True
        else:
            logger.info("✅ Dados estáveis. A distribuição se mantém semelhante ao treino. Modelo saudável.")
            return False
            
    except Exception as e:
        logger.error(f"Erro fatal ao extrair JSON do Evidently: {e}.")
        return False

if __name__ == "__main__":
    check_data_drift()