import os

class Settings:
    PROJECT_NAME: str = "Stock Data API"
    VERSION: str = "1.0.0"
    API_PREFIX: str = ""
    
    # AWS
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET_NAME: str = os.getenv("BUCKET_NAME", "sub-challanger-prd-raw-zone-593793061865")
    
    # MLflow
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://54.82.227.100:5000")

    #paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PIPELINE_PATH: str = os.getenv("PIPELINE_PATH", f"{BASE_DIR}\\models\\pipeline_limpeza_V2.pkl")
    MODEL_PATH: str = os.getenv("MODEL_PATH", f"{BASE_DIR}\\models\\melhor_modelo.pkl")
settings = Settings()
