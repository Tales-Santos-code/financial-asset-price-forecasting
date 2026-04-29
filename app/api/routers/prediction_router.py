from fastapi import APIRouter, Query, HTTPException
from app.api.schemas.prediction_schema import PredictionResponse, StockSymbol, StockInterval, StockPeriod
from app.services.finance_service import FinanceService
from app.api.services.prediction import pipe_to_predict
from app.api.core.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()

# Endpoint GET padronizado usando os Schemas
@router.get("/stock-data-prediction", response_model=PredictionResponse)
def predict_stock(
    symbol: StockSymbol = Query(StockSymbol.RACE, description="Ticker da ação (ex: RACE)"),
    interval: StockInterval = Query(StockInterval.ONE_DAY, description="Intervalo das cotações"),
    period: StockPeriod = Query(StockPeriod.HUNDRED_DAYS, description="Janela de histórico necessária")
):
    ticker = symbol.value
    
    try:
        finance_api = FinanceService(ticker=ticker)
        
        # 1. Busca Dados Crus da Ação
        logger.info(f"📥 Buscando histórico para {ticker} ({period.value})...")
        df_history = finance_api.get_historical_data(full=True, period=period.value)
        
        if df_history.empty: 
            raise ValueError(f"Sem dados encontrados para o ticker {ticker}")
        
        # 2. Busca Dados Crus Macroeconômicos
        min_date = df_history.index.min()
        max_date = df_history.index.max()
        df_macro = finance_api.get_macro_data(min_date=min_date, max_date=max_date)

        # 3. Delega o processamento pesado para o Serviço de Predição
        resultado = pipe_to_predict(ticker, df_history, df_macro)
        
        return resultado

    except ValueError as ve:
        # Erros de regra de negócio (ex: ticker não encontrado, pipeline vazio)
        logger.error(f"Erro de Validação: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Erros inesperados (ex: Yahoo Finance fora do ar)
        logger.error(f"Erro Interno no Servidor: {str(e)}")
        raise HTTPException(status_code=500, detail="Ocorreu um erro ao processar a predição.")