from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

# ==========================================
# 1. ENUMS (Validação Restrita)
# ==========================================
class StockSymbol(str, Enum):
    RACE = "RACE"          # Ferrari (Nosso alvo principal)
    AAPL = "AAPL"          # Apple
    NVDA = "NVDA"          # Nvidia
    VALE3 = "VALE3.SA"     # Vale
    ITSA4 = "ITSA4.SA"     # Itaúsa
    WEGE3 = "WEGE3.SA"     # WEG
    GSPC = "^GSPC"         # S&P 500 (Para testes macro)

class StockInterval(str, Enum):
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    SIXTY_MINUTES = "60m"
    ONE_DAY = "1d"         # Padrão para os nossos dados diários
    ONE_WEEK = "1wk"
    ONE_MONTH = "1mo"

class StockPeriod(str, Enum):
    ONE_MONTH = "1mo"
    THREE_MONTHS = "3mo"
    SIX_MONTHS = "6mo"
    HUNDRED_DAYS = "100d"  # Essencial para calcular a Média Móvel de 50 (SMA_50)
    ONE_YEAR = "1y"
    TWO_YEARS = "2y"
    FIVE_YEARS = "5y"
    MAX = "max"

# ==========================================
# 2. SCHEMAS DE REQUEST & RESPONSE
# ==========================================

class PredictionRequest(BaseModel):
    symbol: StockSymbol = Field(default=StockSymbol.RACE, description="Ticker da ação para predição")
    interval: StockInterval = Field(default=StockInterval.ONE_DAY, description="Intervalo entre as velas de cotação")
    period: StockPeriod = Field(default=StockPeriod.HUNDRED_DAYS, description="Janela histórica necessária para gerar as features")
    
    # Opcionais caso queira forçar datas específicas no yfinance
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class PredictionResponse(BaseModel):
    """
    O JSON exato que o seu usuário (ou front-end) vai receber.
    """
    symbol: str = Field(..., description="Ticker consultado")
    current_price: float = Field(..., description="Cotação de fechamento do dia da referência")
    predicted_price_tomorrow: float = Field(..., description="Preço previsto para o próximo dia útil (XGBoost)")
    variation_pct: float = Field(..., description="Variação percentual prevista em relação ao preço atual")
    timestamp: str = Field(..., description="Data da última cotação real utilizada como base")
    
    # Mensagem opcional de sistema
    message: Optional[str] = None