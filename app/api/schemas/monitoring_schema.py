from pydantic import BaseModel
from typing import Optional

class DriftStatusResponse(BaseModel):
    status: str
    dataset_drift_detected: bool
    last_check_timestamp: str
    message: str

class ModelHealthResponse(BaseModel):
    model_version: str
    is_online: bool
    total_predictions_today: Optional[int] = 0