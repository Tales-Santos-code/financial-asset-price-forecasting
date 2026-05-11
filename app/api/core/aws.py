import boto3
import os
from app.api.core.config import settings
from app.api.core.logger import setup_logger

logger = setup_logger("aws_core")

_session_cache = None
_s3_client_cache = None

def get_aws_session() -> boto3.Session:
    global _session_cache
    if _session_cache is not None:
        return _session_cache
        
    try:
        region = getattr(settings, "AWS_REGION", "us-east-1")
        if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
            logger.info("Usando IAM Role nativo (Lambda).")
            _session_cache = boto3.Session(region_name=region)
        else:
            access_key = (settings.AWS_ACCESS_KEY_ID or "").strip()
            secret_key = (settings.AWS_SECRET_ACCESS_KEY or "").strip()
            if access_key and secret_key:
                _session_cache = boto3.Session(
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    region_name=region
                )
            else:
                _session_cache = boto3.Session(region_name=region)
        return _session_cache
    except Exception as e:
        logger.error(f"Erro na sessão AWS: {e}")
        raise

def get_s3_client():
    global _s3_client_cache
    if _s3_client_cache is not None:
        return _s3_client_cache
        
    session = get_aws_session()
    _s3_client_cache = session.client("s3")
    return _s3_client_cache
