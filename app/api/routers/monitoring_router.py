from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import HTMLResponse
from datetime import datetime

# Presumi que TriggerDriftCheck seja um Enum. Se for, você precisa adicionar "ALL" a ele, 
# ou usar apenas 'str' como fiz abaixo para maior flexibilidade.
from app.api.schemas.monitoring_schema import DriftStatusResponse, ModelHealthResponse
from app.api.core.config import settings
from app.api.core.logger import setup_logger
from app.api.services.drift_detector import check_data_drift
from app.api.services.s3 import read_html_from_s3

logger = setup_logger(__name__)
router = APIRouter()

# Lista de todos os ativos que você monitora. 
# DICA: No futuro, o ideal é puxar isso de um banco de dados ou do settings!
ATIVOS_MONITORADOS = ["RACE", "AAPL", "NVDA", "VALE3.SA", "ITSA4.SA", "WEGE3.SA", "^GSPC"]

@router.get("/health", response_model=ModelHealthResponse)
def check_model_health():
    """
    Endpoint rápido para o Load Balancer da AWS saber se o modelo está vivo.
    """
    return ModelHealthResponse(
        model_version="XGBoost_Ferrari_v2.1",
        is_online=True,
        total_predictions_today=42 
    )

@router.post("/trigger-drift-check", response_model=DriftStatusResponse)
def trigger_drift_analysis(
    background_tasks: BackgroundTasks,
    symbol: str = Query("ALL", description="Ticker da ação (ex: RACE) ou 'ALL' para rodar todos.")
):
    """
    Força a execução do Evidently AI.
    Se 'ALL' for passado, processa todos os ativos da carteira em paralelo.
    """
    ticker = symbol.upper()
    
    try:
        if ticker == "ALL":
            logger.info("🔄 Solicitação de análise de Drift em LOTE (ALL) recebida.")
            
            # Dispara uma task em background separada para cada ativo
            for ativo in ATIVOS_MONITORADOS:
                background_tasks.add_task(check_data_drift, ativo)
            
            return DriftStatusResponse(
                status="Processando Lote",
                dataset_drift_detected=False,
                last_check_timestamp=datetime.utcnow().isoformat(),
                message=f"Análise em background iniciada para {len(ATIVOS_MONITORADOS)} ativos simultaneamente."
            )
            
        else:
            logger.info(f"🔍 Solicitação de análise individual recebida para {ticker}.")
            
            # Roda apenas o ticker específico solicitado
            background_tasks.add_task(check_data_drift, ticker)
            
            return DriftStatusResponse(
                status="Processando Individual",
                dataset_drift_detected=False,
                last_check_timestamp=datetime.utcnow().isoformat(),
                message=f"Análise iniciada em background para {ticker}."
            )

    except Exception as e:
        logger.error(f"❌ Erro ao acionar drift check: {str(e)}")
        raise HTTPException(status_code=500, detail="Falha ao iniciar o detector de drift.")
    
    
@router.get("/drift-report/{symbol}", response_class=HTMLResponse, summary="Visualizar Dashboard")
def view_drift_report(symbol: str):
    """
    Busca o dashboard HTML gerado pelo Evidently AI diretamente do Data Lake (S3).
    """
    ticker = symbol.upper()
    s3_key = f"monitoring/drift_reports/drift_report_{ticker}.html"
    
    html_content = read_html_from_s3(settings.S3_BUCKET_NAME, s3_key)
    
    if not html_content:
        logger.warning(f"Tentativa de acessar relatório inexistente no S3 para {ticker}.")
        raise HTTPException(
            status_code=404, 
            detail=f"Relatório não encontrado no Data Lake para {ticker}."
        )
        
    logger.info(f"Servindo dashboard de Data Drift via S3 para {ticker}.")
    return HTMLResponse(content=html_content, status_code=200)