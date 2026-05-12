import os
import pandas as pd
import requests 

from app.api.core.config import settings
from app.api.core.logger import setup_logger

# Importa todos os conectores necessários do S3
from app.api.services.s3 import write_html_to_s3, read_json_from_s3, read_csv_from_s3
from app.api.core.aws import get_s3_client
from evidently import Report
from evidently.presets import DataDriftPreset

logger = setup_logger("drift_detector")

def disparar_retreino_github(symbol: str):
    """
    Bate na API do GitHub Actions para iniciar o workflow de retreino.
    """
    logger.info(f"🚀 Acionando GitHub Actions para iniciar retreino de {symbol}...")
    
    # Busca as variáveis injetadas na AWS Lambda (com fallback para settings caso rode local)
    github_token = os.environ.get("GITHUB_TOKEN") or getattr(settings, "GITHUB_TOKEN", None)
    github_owner = os.environ.get("GITHUB_OWNER") or getattr(settings, "GITHUB_OWNER", "SEU_USUARIO_GITHUB")
    github_repo = os.environ.get("GITHUB_REPO") or getattr(settings, "GITHUB_REPO", "NOME_DO_SEU_REPOSITORIO")
    
    if not github_token:
        logger.error("❌ GITHUB_TOKEN não encontrado nas variáveis de ambiente da Lambda! O retreino não será acionado.")
        return
        
    url = f"https://api.github.com/repos/{github_owner}/{github_repo}/actions/workflows/retreino_workflow.yml/dispatches"
    
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {github_token}"
    }
    
    payload = {
        "ref": "main", # Branch onde está o arquivo yml
        "inputs": {"symbol": symbol}
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 204:
            logger.info("✅ Sinal de retreino enviado com sucesso para o GitHub Actions!")
        else:
            logger.error(f"❌ Falha ao acionar GitHub: Status {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"❌ Erro de conexão com a API do GitHub: {e}")


def load_production_logs(symbol: str) -> pd.DataFrame:
    """
    Lê os JSONs gerados pela API de predição DIRETAMENTE DO S3, sem tocar no disco.
    """
    s3_client = get_s3_client()
    bucket = settings.S3_BUCKET_NAME
    prefix = f"predictions/{symbol}/"
    
    records = []
    try:
        # Lista os objetos e ordena por data
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        
        if 'Contents' not in response:
            logger.info("Nenhum log de predição encontrado no S3 para análise.")
            return pd.DataFrame()
            
        # Ordena por LastModified decrescente
        sorted_contents = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
        
        # Limita a leitura aos últimos 100 logs para evitar timeout na Lambda
        LIMIT = 100
        count = 0
        
        for obj in sorted_contents:
            if count >= LIMIT:
                break
                
            if obj['Key'].endswith('.json'):
                data = read_json_from_s3(bucket, obj['Key'])
                if data and "features_input" in data:
                    records.append(data["features_input"])
                    count += 1
                    
    except Exception as e:
        logger.warning(f"Erro ao buscar logs de produção no S3: {e}")
        
    return pd.DataFrame(records)

def check_data_drift(symbol: str) -> bool:
    """
    1. Lê dados históricos do S3 (Referência)
    2. Lê logs de predição do S3 (Current)
    3. Gera relatório de Drift em memória (Evidently 0.7+)
    4. Salva o HTML direto no S3
    """
    symbol = symbol.upper().strip()
    logger.info(f"Iniciando análise de Data Drift para {symbol} (Cloud Native)...")
    
    # ==========================================
    # 1. COLETA DE DADOS (DATA LAKE)
    # ==========================================
    reference_data = read_csv_from_s3(
        settings.S3_BUCKET_NAME, 
        f"data/processed/{symbol}_historical_cleaned.csv"
    )
    
    if reference_data is None or reference_data.empty:
        logger.error(f"Arquivo de referência não encontrado no S3 para {symbol}")
        return False
        
    current_data = load_production_logs(symbol)
    
    if current_data.empty or len(current_data) < 5:
        logger.info(f"Sem volume de dados suficiente em produção para {symbol} (Encontrado: {len(current_data)} logs).")
        return False

    # Remove colunas que não são features (Target, Date e colunas de índice do pandas)
    features_para_analisar = [
        c for c in reference_data.columns 
        if c not in ['Target_Log_Return', 'Date'] and not c.startswith('Unnamed')
    ]
    
    for col in features_para_analisar:
        if col not in current_data.columns:
            # Preenche com a média da referência para evitar o 'RuntimeWarning' de divisão por zero do NumPy
            current_data[col] = reference_data[col].mean() if not reference_data[col].empty else 0.0

    # ==========================================
    # ARQUITETURA IN-MEMORY PARA AWS (EVIDENTLY V0.7+)
    # Silencia o joblib warning no Lambda
    # Silencia o joblib warning no Lambda (usa /tmp em vez de shared memory inexistente)
    os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

    drift_report = Report([DataDriftPreset()])
    
    logger.info("Calculando distribuições estatísticas nas variáveis...")
    
    # 1. O .run() devolve um objeto SNAPSHOT
    resultado_eval = drift_report.run(
        reference_data=reference_data[features_para_analisar], 
        current_data=current_data[features_para_analisar]
    )

    # 2. Gera o HTML diretamente em memória (String)
    html_content = resultado_eval.get_html_str(as_iframe=False)
    
    # 3. Salva relatório no S3
    s3_key = f"monitoring/drift_reports/drift_report_{symbol}.html"
    write_html_to_s3(settings.S3_BUCKET_NAME, s3_key, html_content)
    
    logger.info(f"Dashboard injetado direto no Data Lake: s3://{settings.S3_BUCKET_NAME}/{s3_key}")
    
    # ==========================================
    # 5. Extrai os metadados do Evidently
    # ==========================================
    try:
        report_dict = resultado_eval.dict()
        
        dataset_drift = False 
        
        # O Dataset Drift global é calculado pela métrica 'DriftedColumnsCount'
        for metric in report_dict.get("metrics", []):
            m_config = metric.get("config", {})
            m_value = metric.get("value", {})
            m_type = m_config.get("type", "")
            
            # 1. Identifica a métrica global de contagem de colunas com drift
            if "DriftedColumnsCount" in m_type:
                drift_share_threshold = m_config.get("drift_share", 0.5)
                actual_share = m_value.get("share", 0.0)
                
                if actual_share >= drift_share_threshold:
                    dataset_drift = True
                    logger.warning(f"🚨 Drift Global detectado: {actual_share:.2%} de colunas afetadas (Threshold: {drift_share_threshold:.2%})")
                else:
                    logger.info(f"✅ Drift Global abaixo do threshold: {actual_share:.2%} de colunas (Threshold: {drift_share_threshold:.2%})")
                break
        
        if dataset_drift:
            logger.warning(f"⚠️ DRIFT CONFIRMADO para {symbol}! Disparando retreino no GitHub Actions...")
            disparar_retreino_github(symbol)
            return True
        else:
            logger.info(f"✅ Sem drift significativo detectado para {symbol}.")
            return False
            
    except Exception as e:
        logger.error(f"Erro fatal ao extrair dados do Evidently: {e}")
        return False

if __name__ == "__main__":
    check_data_drift("RACE")