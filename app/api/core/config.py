from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from pathlib import Path

# Define a raiz do projeto para fallback e arquivos temporários (se necessário)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "Stock Data API"
    VERSION: str = "1.0.0"
    API_PREFIX: str = ""
    
    # AWS
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str = "financial-asset-price-forecasting-495599733085-us-east-1-an"
    
    # MLflow
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://54.82.227.100:5000")

    # github
    GITHUB_TOKEN: str
    GITHUB_OWNER: str
    GITHUB_REPO: str

    # paths (não mais utilizados para carregar modelos de forma hardcoded, os modelos vêm do S3)
    BASE_DIR: Path = BASE_DIR
    PIPELINE_PATH: str = os.getenv("PIPELINE_PATH", str(BASE_DIR / "models" / "pipeline_limpeza_V2.pkl"))
    MODEL_PATH: str = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "champions" / "melhor_modelo.pkl"))

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
