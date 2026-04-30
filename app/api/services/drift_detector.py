import os
import json
import glob
import pandas as pd

# ==========================================
# NOVAS IMPORTAÇÕES DO EVIDENTLY (v0.7+)
# ==========================================
from evidently import Report
from evidently.presets import DataDriftPreset

from app.api.core.config import settings
from app.api.core.logger import setup_logger

logger = setup_logger("drift_detector")

def load_production_logs(symbol: str) -> pd.DataFrame:
    """
    Função auxiliar que lê os JSONs gerados pela API de predição.
    """
    log_path = os.path.join(settings.BASE_DIR, "data", "predictions", symbol, "*.json")
    log_files = glob.glob(log_path)
    
    if not log_files:
        return pd.DataFrame()
        
    records = []
    for file in log_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                features = data.get("features_input", {})
                records.append(features)
        except Exception as e:
            logger.warning(f"Erro ao ler log {file}: {e}")
            
    return pd.DataFrame(records)


def check_data_drift(symbol: str = "RACE") -> bool:
    """
    Compara o CSV de treinamento original com os logs de produção recentes.
    """
    logger.info(f"🔍 Iniciando análise de Data Drift (Evidently v0.7+) para {symbol}...")
    
    ref_path = os.path.join(settings.BASE_DIR, 'app', 'ml', 'pipeline', 'dados_transformados_V2.csv')
    if not os.path.exists(ref_path):
        logger.error(f"Arquivo de referência (treinamento) não encontrado. \n {ref_path}")
        return False
        
    reference_data = pd.read_csv(ref_path)
    current_data = load_production_logs(symbol)
    
    if current_data.empty or len(current_data) < 5:
        logger.info("⚠️ Sem volume de dados suficiente em produção para analisar Drift estatístico.")
        return False

    features_para_analisar = [c for c in reference_data.columns if c not in ['Target_Log_Return', 'Date']]
    
    for col in features_para_analisar:
        if col not in current_data.columns:
            current_data[col] = 0.0

    # ==========================================
    # A NOVA ARQUITETURA DE EXECUÇÃO
    # ==========================================
    # 1. Instancia o Report recebendo os presets diretamente numa lista
    drift_report = Report([DataDriftPreset()])
    
    logger.info("⚙️ Calculando distribuições estatísticas nas variáveis...")
    
    # 2. O .run() agora retorna o objeto de Snapshot!
    resultado_eval = drift_report.run(
        reference_data=reference_data[features_para_analisar], 
        current_data=current_data[features_para_analisar]
    )
    
    report_path = os.path.join(settings.BASE_DIR, "data", f"drift_report_{symbol}.html")
    os.makedirs(os.path.dirname(report_path), exist_ok=True) 
    
    # 3. Chama o HTML a partir do Snapshot, e não do Report
    resultado_eval.save_html(report_path)
    logger.info(f"📊 Dashboard visual salvo em: {report_path}")
    
    # 4. Extrai o JSON a partir do Snapshot
    report_json = resultado_eval.dict()
    dataset_drift = report_json["metrics"][0]["result"]["dataset_drift"]
    
    if dataset_drift:
        logger.warning("🚨 ALERTA VERMELHO: Data Drift Detectado! O comportamento do mercado mudou.")
        return True
    else:
        logger.info("✅ Dados estáveis. A distribuição se mantém semelhante ao treino. Modelo saudável.")
        return False

if __name__ == "__main__":
    check_data_drift()