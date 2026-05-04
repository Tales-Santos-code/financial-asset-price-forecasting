import boto3
from app.api.core.config import settings
from app.api.core.logger import setup_logger

logger = setup_logger("aws_core")

def get_aws_session() -> boto3.Session:
    """
    Cria e retorna uma sessão centralizada da AWS utilizando as 
    credenciais seguras carregadas do .env (via Pydantic Settings).
    """
    try:
        session = boto3.Session(
            region_name=settings.AWS_REGION
        )
        logger.info("Sessão AWS criada com sucesso.")
        return session 
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