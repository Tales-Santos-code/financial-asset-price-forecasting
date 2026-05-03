from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import HTMLResponse
from datetime import datetime

from app.api.schemas.monitoring_schema import DriftStatusResponse, ModelHealthResponse
from app.api.core.config import settings
from app.api.core.logger import setup_logger
from app.api.services.drift_detector import check_data_drift
from app.api.services.s3 import read_html_from_s3

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
def trigger_drift_analysis(
    background_tasks: BackgroundTasks,
    symbol: str = Query("RACE", description="Ticker da ação para analisar drift")
):
    """
    Força a execução do Evidently AI.
    Processamento In-Memory rodando em background (Cloud Native).
    """
    ticker = symbol.upper()
    try:
        logger.info(f"Solicitação de análise de Data Drift recebida via API para {ticker}.")
        
        # Manda a função pesada rodar em segundo plano passando o ticker escolhido
        background_tasks.add_task(check_data_drift, ticker)
        
        return DriftStatusResponse(
            status="Processando",
            dataset_drift_detected=False,
            last_check_timestamp=datetime.utcnow().isoformat(),
            message=f"Análise do Evidently AI iniciada em segundo plano para {ticker}. O HTML será enviado ao S3."
        )

    except Exception as e:
        logger.error(f"Erro ao acionar drift check: {str(e)}")
        raise HTTPException(status_code=500, detail="Falha ao iniciar o detector de drift.")
    
    
@router.get("/drift-report/{symbol}", response_class=HTMLResponse, summary="Visualizar Dashboard de Data Drift")
def view_drift_report(symbol: str):
    """
    Busca o dashboard HTML gerado pelo Evidently AI diretamente do Data Lake (S3).
    """
    ticker = symbol.upper()
    s3_key = f"monitoring/drift_reports/drift_report_{ticker}.html"
    
    # Busca a string de HTML puro 100% da AWS
    html_content = read_html_from_s3(settings.S3_BUCKET_NAME, s3_key)
    
    if not html_content:
        logger.warning(f"Tentativa de acessar relatório inexistente no S3 para {ticker}.")
        raise HTTPException(
            status_code=404, 
            detail=f"Relatório não encontrado no Data Lake para {ticker}. Rode a predição e force a análise primeiro."
        )
        
    logger.info(f"Servindo dashboard de Data Drift via S3 para {ticker}.")
    return HTMLResponse(content=html_content, status_code=200)