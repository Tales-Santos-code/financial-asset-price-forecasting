from fastapi import APIRouter, Query, HTTPException, BackgroundTasks
from app.api.schemas.prediction_schema import PredictionResponse, StockSymbol
from app.api.services.finance_service import FinanceService
from app.api.services.prediction import pipe_to_predict
from app.api.core.logger import setup_logger
from app.api.services.s3 import read_csv_from_s3, write_csv_to_s3
from app.api.core.config import settings
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

logger = setup_logger(__name__)
router = APIRouter()

def _update_master_history_bg(bucket, key, df):
    """Tarefa de fundo para não travar a resposta do usuário"""
    try:
        df_to_save = df.reset_index()
        if 'index' in df_to_save.columns:
            df_to_save.rename(columns={'index': 'Date'}, inplace=True)
        write_csv_to_s3(bucket, key, df_to_save)
        logger.info(f"💾 [BG] Histórico master atualizado no S3 ({len(df)} linhas).")
    except Exception as e:
        logger.error(f"Erro ao atualizar master history em background: {e}")

@router.get("/stock-data-prediction", response_model=PredictionResponse)
def predict_stock(
    background_tasks: BackgroundTasks,
    symbol: str = Query("RACE", description="Ticker da ação (ex: RACE, AAPL, NVDA)"),
):
    ticker = symbol.upper().strip()
    tickers_permitidos = [item.value for item in StockSymbol]
    
    if ticker not in tickers_permitidos:
        raise HTTPException(status_code=400, detail=f"Ticker '{ticker}' não suportado.")
    
    try:
        finance_api = FinanceService(ticker=ticker)
        bucket = settings.S3_BUCKET_NAME
        historico_key = f"data/historical/{ticker}_master_history.csv"
        
        logger.info(f"🚀 Iniciando coleta paralela para {ticker}...")

        # PARALELISMO: Busca S3, Yahoo e Macro tudo junto!
        with ThreadPoolExecutor(max_workers=3) as executor:
            f_novos = executor.submit(finance_api.get_historical_data, full=True, use_checkpoint=True)
            f_antigo = executor.submit(read_csv_from_s3, bucket, historico_key)
            
            # Para ganhar tempo, vamos assumir que o macro quer os últimos 200 dias de hoje
            now = pd.Timestamp.now()
            f_macro = executor.submit(finance_api.get_macro_data, 
                                     min_date=(now - pd.Timedelta(days=200)).strftime('%Y-%m-%d'),
                                     max_date=now.strftime('%Y-%m-%d'))

        df_novos = f_novos.result()
        df_antigo = f_antigo.result()
        df_macro = f_macro.result()

        # Consolidação rápida
        df_history = pd.DataFrame()
        if df_antigo is not None and not df_antigo.empty:
            if 'Date' in df_antigo.columns:
                df_antigo['Date'] = pd.to_datetime(df_antigo['Date'])
                df_antigo.set_index('Date', inplace=True)
            df_history = df_antigo
        
        if not df_novos.empty:
            df_history = pd.concat([df_history, df_novos])
            df_history = df_history[~df_history.index.duplicated(keep='last')].sort_index()
            
            # Agenda a atualização do S3 para DEPOIS de responder o usuário
            background_tasks.add_task(_update_master_history_bg, bucket, historico_key, df_history)

        if df_history.empty: 
            raise ValueError(f"Sem dados para {ticker}")
        
        # OTIMIZAÇÃO: O pipeline só precisa de ~200 dias para calcular médias móveis pesadas (ex: SMA 200)
        # Enviar 6.000 linhas mata a performance da Lambda.
        df_ml_input = df_history.tail(250).copy()
        
        # Predição final
        resultado = pipe_to_predict(ticker, df_ml_input, df_macro)
        return resultado

    except ValueError as ve:
        logger.error(f"Erro de Validação: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Erro na predição: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")