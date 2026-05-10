import boto3
import os
from app.api.core.config import settings
from app.api.core.logger import setup_logger

logger = setup_logger("aws_core")

def get_aws_session() -> boto3.Session:
    """
    Cria e retorna uma sessão centralizada da AWS utilizando as 
    credenciais seguras carregadas do .env (via Pydantic Settings).
    """
    try:
        if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
            # No Lambda, a melhor prática é NÃO passar credenciais.
            # O Boto3 encontra o IAM Role automaticamente no ambiente.
            logger.info("Execução em AWS Lambda detectada. Usando IAM Role nativo.")
            return boto3.Session(region_name=settings.AWS_REGION)
        
        # Local Dev ou outros ambientes (EC2, Docker local)
        access_key = (settings.AWS_ACCESS_KEY_ID or "").strip()
        secret_key = (settings.AWS_SECRET_ACCESS_KEY or "").strip()

        if access_key and secret_key:
            logger.info(f"Credenciais estáticas detectadas (Local Dev). Key: {access_key[:4]}...")
            return boto3.Session(
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=settings.AWS_REGION
            )
        
        logger.info("Nenhuma credencial configurada. Usando Default Provider Chain.")
        return boto3.Session(region_name=settings.AWS_REGION)

    except Exception as e:
        logger.error(f"Falha ao iniciar sessão AWS: {e}")
        raise

def get_s3_client():
    """
    Retorna um client do S3 pronto para uso.
    """
    session = get_aws_session()
    return session.client("s3")

# Se no futuro você precisar de outro serviço, é só adicionar aqui:
# def get_dynamodb_client(): ...