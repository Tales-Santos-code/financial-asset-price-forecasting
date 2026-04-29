from fastapi import APIRouter, HTTPException, BackgroundTasks
from datetime import datetime
import json
import os

from app.api.schemas.monitoring_schema import DriftStatusResponse, ModelHealthResponse
from app.api.core.config import settings
from app.api.core.logger import setup_logger
from app.api.services.drift_detector import check_data_drift  # Importa a lógica do Evidently AI

logger = setup_logger(__name__)
router = APIRouter()

@router.get("/health", response_model=ModelHealthResponse)
def check_model_health():
    """
    Endpoint rápido para o Load Balancer da AWS saber se o modelo está vivo.
    """
    return ModelHealthResponse(
        model_version="XGBoost_Ferrari_v2.1",
        is_online=True,
        total_predictions_today=42 # Na vida real, viria do banco de dados/S3
    )

@router.post("/trigger-drift-check", response_model=DriftStatusResponse)
def trigger_drift_analysis(background_tasks: BackgroundTasks):
    """
    Força a execução do Evidently AI.
    Como calcular Drift pode demorar (processamento pesado), nós usamos o 
    'BackgroundTasks' do FastAPI para liberar a API imediatamente enquanto ele calcula no fundo.
    """
    try:
        logger.info("Solicitação de análise de Data Drift recebida via API.")
        
        # Manda a função pesada rodar em segundo plano
        background_tasks.add_task(check_data_drift)
        
        return DriftStatusResponse(
            status="Processando",
            dataset_drift_detected=False,
            last_check_timestamp=datetime.utcnow().isoformat(),
            message="Análise do Evidently AI iniciada em segundo plano. Verifique os logs."
        )

    except Exception as e:
        logger.error(f"Erro ao acionar drift check: {str(e)}")
        raise HTTPException(status_code=500, detail="Falha ao iniciar o detector de drift.")