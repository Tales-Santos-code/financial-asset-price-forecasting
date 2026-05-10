from fastapi import APIRouter, Query, HTTPException
from app.api.schemas.prediction_schema import PredictionResponse, StockSymbol
from app.api.services.finance_service import FinanceService
from app.api.services.prediction import pipe_to_predict
from app.api.core.logger import setup_logger
from app.api.services.s3 import read_csv_from_s3, write_csv_to_s3
from app.api.core.config import settings
import pandas as pd

logger = setup_logger(__name__)
router = APIRouter()


# Endpoint GET padronizado usando os Schemas
@router.get("/stock-data-prediction", response_model=PredictionResponse)
def predict_stock(
    # 1. Alterado de StockSymbol para str para capturarmos qualquer texto e tratarmos internamente
    symbol: str = Query("RACE", description="Ticker da ação (ex: RACE, AAPL, NVDA)"),
):
    # 2. Padroniza para maiúsculo (evita erro se o usuário digitar 'race' ou 'aapl' minúsculo)
    ticker = symbol.upper().strip()
    
    # 3. Extrai a lista de valores permitidos dinamicamente do seu Enum
    tickers_permitidos = [item.value for item in StockSymbol]
    
    # 4. A barreira do "Porteiro": Rejeita com erro 400 amigável antes de gastar recursos
    if ticker not in tickers_permitidos:
        logger.warning(f"[REJEITADO] Requisição negada: Ticker '{ticker}' não autorizado.")
        raise HTTPException(
            status_code=400, 
            detail=f"O ticker '{ticker}' não é suportado. Tickers permitidos: {', '.join(tickers_permitidos)}."
        )
    
    try:
        finance_api = FinanceService(ticker=ticker)
        bucket = settings.S3_BUCKET_NAME
        historico_key = f"data/historical/{ticker}_master_history.csv"
        
        # 1. Busca Dados Crus da Ação
        logger.info(f"[DOWNLOAD] Buscando histórico para {ticker} ...")

        df_novos = finance_api.get_historical_data(full=True, use_checkpoint=True)
        df_history = pd.DataFrame()


        # Tenta carregar o histórico Master antigo do S3
        try:
            df_antigo = read_csv_from_s3(bucket, historico_key)
            if df_antigo is not None and not df_antigo.empty:
                # Garante que a coluna de data vire o índice para facilitar o join
                if 'Date' in df_antigo.columns:
                    df_antigo['Date'] = pd.to_datetime(df_antigo['Date'])
                    df_antigo.set_index('Date', inplace=True)
                df_history = df_antigo
        except Exception as e:
            logger.warning(f"Histórico master não encontrado no S3. Criando um novo do zero. ({e})")
        
        if not df_novos.empty:
            df_history = pd.concat([df_history, df_novos])

            # Remove possíveis linhas duplicadas (caso o ponteiro pegue o final do dia anterior)
            # O keep='last' garante que ficaremos com o valor mais atualizado daquela data
            df_history = df_history[~df_history.index.duplicated(keep='last')]
            df_history.sort_index(inplace=True)

            df_to_save = df_history.reset_index()
            if 'index' in df_to_save.columns:
                df_to_save.rename(columns={'index': 'Date'}, inplace=True)

            # Sobrescreve o arquivo Master no S3
            write_csv_to_s3(bucket, historico_key, df_to_save)
            logger.info(f"[OK] Histórico master consolidado no S3. Total de linhas: {len(df_history)}.")


        if df_history.empty: 
            raise ValueError(f"Sem dados encontrados para o ticker {ticker}")
        

        # ==========================================
        # 2. PREPARO DO CONTEXTO PARA O ML
        # ==========================================
        # O Pipeline precisa de histórico para calcular a Média Móvel de 50 dias.
        # Pegamos as últimas 150 linhas do arquivo gigante para garantir matemática perfeita
        # sem sobrecarregar a memória do modelo de ML.
        df_ml_context = df_history.tail(150).copy()
        
        # ==========================================
        # 3. BUSCA MACROECONÔMICA E PREDICÃO
        # ==========================================
        min_date = df_ml_context.index.min()
        max_date = df_ml_context.index.max()
        df_macro = finance_api.get_macro_data(min_date=min_date, max_date=max_date)

        
        # 3. Delega o processamento pesado para o Serviço de Predição
        resultado = pipe_to_predict(ticker, df_history, df_macro)
        
        return resultado

    except ValueError as ve:
        # Erros de regra de negócio (ex: pipeline vazio)
        logger.error(f"Erro de Validação: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Erros inesperados (ex: Yahoo Finance fora do ar)
        logger.error(f"Erro no processamento da predição: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro ao processar a predição: {str(e)}")