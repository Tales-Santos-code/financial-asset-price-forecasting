import yfinance as yf
import pandas as pd

class FinanceService:
    def __init__(self, ticker: str = "RACE", tickers_macro: list = ["^GSPC", "^VIX", "EURUSD=X"]):
        self.stock_historical = yf.Ticker(ticker)
        self.ticker_macro = tickers_macro

    def get_historical_data(self, full: bool = False, period: str = "max", interval: str = "1h", prepost: bool = True, actions: bool = True) -> pd.DataFrame:
        """
        Busca dados históricos de uma ação.
        
        Args:
            full (bool): Se True, retorna todos os dados (incluindo dividendos/splits). Se False, apenas colunas OHLCV.
            period (str): Período de tempo (ex: "1y", "max").
            interval (str): Intervalo (ex: "1h", "1d").
            prepost (bool): Incluir pré/pós mercado.
            actions (bool): Incluir dividendos e splits.
            
        Returns:
            pd.DataFrame: DataFrame do Pandas com os dados históricos.
        """
        # Define o intervalo: se for carga total (full), usamos diário por padrão
        current_interval = "1d" if full else interval
        
        # O yfinance já retorna um pd.DataFrame nativamente
        df_financial = self.stock_historical.history(
            period=period, 
            interval=current_interval, 
            prepost=prepost, 
            actions=actions
        )

        if df_financial.empty:
            raise ValueError(f"Ticker {self.stock_historical.ticker} não encontrado ou sem dados.")
        
        if not full:
            # Retorna apenas as colunas principais
            return df_financial[["Open", "High", "Low", "Close", "Volume"]]

        return df_financial
    
    def get_macro_data(self, min_date, max_date) -> pd.DataFrame:
        """
        Busca dados macroeconômicos comparativos.
        """
        # Garante que max_date seja tratado como datetime para a operação de soma
        max_date_dt = pd.to_datetime(max_date)
        
        # Substituído pl.Timedelta por pd.Timedelta
        df_macro = yf.download(
            self.ticker_macro, 
            start=min_date, 
            end=max_date_dt + pd.Timedelta(days=5), 
            progress=False
        )

        return df_macro