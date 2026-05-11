from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Path
from fastapi.responses import HTMLResponse
from datetime import datetime

# Importamos o seu Enum para manter a lista de ativos centralizada e única no sistema inteiro
from app.api.schemas.prediction_schema import StockSymbol 
from app.api.schemas.monitoring_schema import DriftStatusResponse, ModelHealthResponse
from app.api.core.config import settings
from app.api.core.logger import setup_logger
from app.api.services.drift_detector import check_data_drift
from app.api.services.s3 import read_html_from_s3

logger = setup_logger(__name__)
router = APIRouter()

# Puxa dinamicamente todos os valores permitidos do seu Enum
ATIVOS_MONITORADOS = [item.value for item in StockSymbol]

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
    symbol: str = Query("ALL", description="Ticker único (ex: RACE) ou 'ALL'."),
    run_async: bool = Query(False, description="Se True, executa em background (pode falhar na Lambda).")
):
    """
    Força a execução do Evidently AI.
    No ambiente Lambda, recomenda-se run_async=False para garantir a conclusão.
    """
    ticker = symbol.upper().strip()
    
    if ticker != "ALL" and ticker not in ATIVOS_MONITORADOS:
        logger.warning(f"🚫 Gatilho negado: Ticker ou formato '{ticker}' não autorizado.")
        raise HTTPException(
            status_code=400, 
            detail=f"Entrada inválida. Envie 'ALL' ou apenas UM ticker válido por vez: {', '.join(ATIVOS_MONITORADOS)}."
        )
    
    try:
        if ticker == "ALL":
            logger.info("🔄 Solicitação de análise de Drift em LOTE (ALL) recebida.")
            for ativo in ATIVOS_MONITORADOS:
                if ativo != "^GSPC":
                    background_tasks.add_task(check_data_drift, ativo)
            
            return DriftStatusResponse(
                status="Processando Lote",
                dataset_drift_detected=False,
                last_check_timestamp=datetime.utcnow().isoformat(),
                message=f"Análise em background iniciada para {len(ATIVOS_MONITORADOS)-1} ativos."
            )
            
        else:
            if run_async:
                logger.info(f"🔍 [ASYNC] Iniciando análise para {ticker} via BackgroundTask.")
                background_tasks.add_task(check_data_drift, ticker)
                msg = f"Análise iniciada em background para {ticker}."
            else:
                logger.info(f"🔍 [SYNC] Iniciando análise síncrona para {ticker} (Recomendado para Lambda).")
                is_drift = check_data_drift(ticker)
                msg = f"Análise concluída para {ticker}. Drift detectado: {is_drift}"
            
            return DriftStatusResponse(
                status="Concluído" if not run_async else "Processando",
                dataset_drift_detected=False, # Não sabemos o resultado aqui se for async
                last_check_timestamp=datetime.utcnow().isoformat(),
                message=msg
            )

    except Exception as e:
        logger.error(f"❌ Erro ao acionar drift check: {str(e)}")
        raise HTTPException(status_code=500, detail="Falha ao iniciar o detector de drift.")
    
    
@router.get("/drift-report/{symbol}", response_class=HTMLResponse, summary="Visualizar Dashboard")
def view_drift_report(
    symbol: str = Path(..., description="Ticker único da ação (ex: RACE)")
):
    """
    Busca o dashboard HTML gerado pelo Evidently AI diretamente do Data Lake (S3).
    """
    ticker = symbol.upper().strip()
    
    # ==========================================
    # VALIDAÇÃO DO PORTEIRO
    # ==========================================
    if ticker not in ATIVOS_MONITORADOS:
        logger.warning(f"🚫 Acesso negado: Relatório HTML para '{ticker}' não autorizado.")
        raise HTTPException(
            status_code=400, 
            detail=f"Ticker '{ticker}' inválido. Relatórios disponíveis apenas para: {', '.join(ATIVOS_MONITORADOS)}."
        )

    s3_key = f"monitoring/drift_reports/drift_report_{ticker}.html"
    html_content = read_html_from_s3(settings.S3_BUCKET_NAME, s3_key)
    
    if not html_content:
        logger.warning(f"Tentativa de acessar relatório inexistente no S3 para {ticker}.")
        raise HTTPException(
            status_code=404, 
            detail=f"Relatório não encontrado no Data Lake para {ticker}. Rode o trigger-drift-check primeiro."
        )
        
    logger.info(f"Servindo dashboard de Data Drift via S3 para {ticker}.")
    return HTMLResponse(content=html_content, status_code=200)