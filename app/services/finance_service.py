import yfinance as yf
import polars as pl

class FinanceService:

    def get_historical_data(self, full:bool = False, ticker: str = "RACE", period: str = "max", interval: str = "1h", prepost: bool = True, actions: bool = True) -> pl.DataFrame | tuple[pl.DataFrame, dict, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Busca dados históricos de uma ação.
        Args:
            ticker (str): O símbolo da ação (ex: "AAPL" para Apple).
            period (str): O período de tempo para os dados históricos (ex: "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max").
            interval (str): O intervalo de tempo entre os dados (ex: "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d").
            prepost (bool): Incluir dados pré-mercado e pós-mercado.
            actions (bool): Incluir ações corporativas como dividendos e splits.
        Returns:
            df: pl.DataFrame: Um DataFrame contendo os dados históricos da ação.
            metadados: dict: Um dicionário contendo os metadados da ação.
            financeiro: pl.DataFrame: Um DataFrame contendo os dados financeiros da ação.
            balanço_patrimonial: pl.DataFrame: Um DataFrame contendo o balanço patrimonial da ação.
            fluxo_de_caixa: pl.DataFrame: Um DataFrame contendo o fluxo de caixa da ação.
        Raises:
            ValueError: Se o ticker não for encontrado ou não tiver dados.  
        """

        stock = yf.Ticker(ticker)
        current_interval = "1d" if full else interval
        df = stock.history(period=period, interval=current_interval, prepost=prepost, actions=actions)

        if df.empty:
            raise ValueError(f"Ticker {ticker} não encontrado ou sem dados.")
        
        if not full:
            return df[["Open", "High", "Low", "Close", "Volume"]]
        
        metadados = stock.info

        financeiro = stock.financials
        balanço_patrimonial = stock.balance_sheet
        fluxo_de_caixa = stock.cashflow

        
        return df, metadados, financeiro, balanço_patrimonial, fluxo_de_caixa
