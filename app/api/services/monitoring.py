from datetime import datetime
import pandas as pd
from app.api.core.logger import setup_logger
from app.api.core.config import settings
# from app.api.services.s3 import write_json_to_s3 # (Assumindo que você tem essa abstração do boto3)

import os
import json

from app.api.services.s3 import write_json_to_s3

# Ajustado para usar o setup_logger do seu core
logger = setup_logger("monitoring_service")

def save_prediction_log(symbol: str, features_utilizadas: pd.DataFrame, prediction_result: dict) -> None:
    """
    Salva os detalhes da predição e as features usadas no S3 para observabilidade.
    Path: s3://<bucket>/predictions/<symbol>/<timestamp>.json
    """
    try:
        timestamp_atual = datetime.utcnow()
        # Formato seguro para nome de arquivo no S3
        filename = timestamp_atual.strftime("%Y-%m-%dT%H-%M-%S-%f")
        key = f"predictions/{symbol}/{filename}.json"
        
        # Converte a única linha do DataFrame (features) para um dicionário puro
        features_dict = features_utilizadas.to_dict(orient='records')[0] if not features_utilizadas.empty else {}
        
        # O Payload completo para Observabilidade (Features + Predição)
        payload = {
            "symbol": symbol,
            "timestamp": timestamp_atual.isoformat(),
            "model_version": "champion", 
            "features_input": features_dict,
            "prediction_output": prediction_result
        }
        
        # Salva no Data Lake
        write_json_to_s3(settings.S3_BUCKET_NAME, key, payload)

        try:
            local_dir = os.path.join(settings.BASE_DIR, "data", "predictions", symbol)
            os.makedirs(local_dir, exist_ok=True)
            
            local_path = os.path.join(local_dir, f"{filename}.json")
            
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=4)
        
            logger.info(f"Log salvo localmente para análise de Drift: {local_path}")
        except Exception as e:
            logger.error(f"Falha catastrófica ao salvar log de monitoramento local: {e}")



        logger.info(f"Log de predição e features salvo no S3: {key}")
        
    except Exception as e:
        logger.error(f"Falha ao salvar log de monitoramento no S3: {e}", exc_info=True)
        # O erro é engolido aqui propositalmente. 
        # Se o S3 falhar, o usuário ainda deve receber a cotação na API.