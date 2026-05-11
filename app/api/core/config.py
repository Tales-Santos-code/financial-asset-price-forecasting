from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from pathlib import Path

# Define a raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "Stock Data API"
    VERSION: str = "1.0.0"
    API_PREFIX: str = ""
    
    # AWS (Opcionais se rodando com IAM Role no Lambda/EC2)
    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str = "financial-asset-price-forecasting-495599733085-us-east-1-an"
    
    # MLflow
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

    # github (Opcionais se não for usar integração direta no Lambda)
    GITHUB_TOKEN: str | None = None
    GITHUB_OWNER: str | None = None
    GITHUB_REPO: str | None = None

    # paths
    BASE_DIR: Path = BASE_DIR
    PIPELINE_PATH: str = os.getenv("PIPELINE_PATH", str(BASE_DIR / "models" / "pipeline_limpeza_V2.pkl"))
    MODEL_PATH: str = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "champions" / "melhor_modelo.pkl"))

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
