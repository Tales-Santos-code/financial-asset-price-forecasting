import yfinance as yf
import pandas as pd
from app.api.core.config import settings
from app.api.core.logger import setup_logger

# Importamos as funções que você já criou no s3.py
from app.api.services.s3 import read_json_from_s3, write_json_to_s3
logger = setup_logger("finance_service")

class FinanceService:
    def __init__(self, ticker: str = "RACE", tickers_macro: list = ["^GSPC", "^VIX", "EURUSD=X"]):
        self.ticker_symbol = ticker # Armazenamos a string para usar no nome do arquivo no S3
        self.stock_historical = yf.Ticker(ticker)
        self.ticker_macro = tickers_macro

    def get_historical_data(self, full: bool = False, interval: str = "1d", prepost: bool = True, actions: bool = False, use_checkpoint: bool = True) -> pd.DataFrame:
        """
        Busca dados históricos de uma ação de forma inteligente (incremental).
        """
        current_interval = "1d" if full else interval
        bucket = settings.S3_BUCKET_NAME
        
        # O caminho do nosso "Ponteiro" no S3
        pointer_key = f"checkpoints/{self.ticker_symbol}_{current_interval}_pointer.json"
        start_date = None

        if use_checkpoint:
            # 1. Busca o ponteiro no S3
            pointer_data = read_json_from_s3(bucket, pointer_key)
            
            if pointer_data and "last_date" in pointer_data:
                # Pega a última data e soma 1 dia (para não baixar o último dia duplicado)
                last_date_dt = pd.to_datetime(pointer_data["last_date"])
                start_date = (last_date_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                logger.info(f"Ponteiro encontrado! Baixando dados APENAS a partir de {start_date}")
            else:
                logger.info("Nenhum ponteiro encontrado no S3. Iniciando carga total (max).")

        # 2. Chama a API do Yahoo Finance com a inteligência aplicada
        if start_date:
            df_financial = self.stock_historical.history(
                start=start_date, 
                interval=current_interval, 
                prepost=prepost, 
                actions=actions
            )
        else:
            df_financial = self.stock_historical.history(
                period="max", 
                interval=current_interval, 
                prepost=prepost, 
                actions=actions
            )

        # 3. Validações e Atualização do Ponteiro
        if df_financial.empty:
            logger.info(f"Nenhum dado novo encontrado para {self.ticker_symbol} na API.")
            return df_financial # Retorna vazio, não há o que atualizar

        # O yfinance traz as datas com fuso horário embutido. Removemos para evitar bugs no Pandas.
        df_financial.index = pd.to_datetime(df_financial.index).tz_localize(None)

        if use_checkpoint:
            # Pega a data mais recente do lote que acabamos de baixar
            new_last_date = df_financial.index.max().strftime("%Y-%m-%d")
            
            # Salva o novo ponteiro no S3
            write_json_to_s3(bucket, pointer_key, {"last_date": new_last_date})
            logger.info(f"Ponteiro atualizado no S3 para: {new_last_date}")

        if not full:
            return df_financial[["Open", "High", "Low", "Close", "Volume"]]

        return df_financial
    
    def get_macro_data(self, min_date, max_date) -> pd.DataFrame:
        """
        Busca dados macroeconômicos comparativos baseado nas datas da ação.
        """
        max_date_dt = pd.to_datetime(max_date)
        
        df_macro = yf.download(
            self.ticker_macro, 
            start=min_date, 
            end=max_date_dt + pd.Timedelta(days=5), 
            progress=False
        )

        return df_macro