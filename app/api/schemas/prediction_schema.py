from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

# ==========================================
# 1. ENUMS (Validação Restrita)
# ==========================================
class StockSymbol(str, Enum):
    RACE = "RACE"          # Ferrari
    AAPL = "AAPL"          # Apple
    NVDA = "NVDA"          # Nvidia
    VALE3 = "VALE3.SA"     # Vale
    ITSA4 = "ITSA4.SA"     # Itaúsa
    WEGE3 = "WEGE3.SA"     # WEG
    GSPC = "^GSPC"         # S&P 500


# ==========================================
# 2. SCHEMAS DE REQUEST & RESPONSE
# ==========================================

class PredictionRequest(BaseModel):
    symbol: StockSymbol = Field(default=StockSymbol.RACE, description="Ticker da ação para predição")
    
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class PredictionResponse(BaseModel):
    """
    O JSON exato que vai receber.
    """
    symbol: str = Field(..., description="Ticker consultado")
    current_price: float = Field(..., description="Cotação de fechamento do dia da referência")
    predicted_price_tomorrow: Optional[float] = Field(None, description="Preço previsto para o próximo dia útil (XGBoost)")
    variation_pct: Optional[float] = Field(None, description="Variação percentual prevista em relação ao preço atual")
    timestamp: str = Field(..., description="Data da última cotação real utilizada como base")
    
    # Mensagem opcional de sistema
    message: Optional[str] = None