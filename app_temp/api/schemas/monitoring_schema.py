from pydantic import BaseModel
from typing import Optional
from enum import Enum

class DriftStatusResponse(BaseModel):
    status: str
    dataset_drift_detected: bool
    last_check_timestamp: str
    message: str

class ModelHealthResponse(BaseModel):
    model_version: str
    is_online: bool
    total_predictions_today: Optional[int] = 0

# ==========================================
# 1. ENUMS (Validação Restrita)
# ==========================================
class TriggerDriftCheck(str, Enum):
    RACE = "RACE"          # Ferrari (Nosso alvo principal)
    AAPL = "AAPL"          # Apple
    NVDA = "NVDA"          # Nvidia
    VALE3 = "VALE3.SA"     # Vale
    ITSA4 = "ITSA4.SA"     # Itaúsa
    WEGE3 = "WEGE3.SA"     # WEG
    GSPC = "^GSPC"         # S&P 500 (Para testes macro)