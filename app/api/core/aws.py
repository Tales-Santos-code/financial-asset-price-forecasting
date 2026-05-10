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
        # Detecta se está rodando no Lambda
        is_lambda = os.environ.get("AWS_LAMBDA_FUNCTION_NAME") is not None
        
        access_key = (settings.AWS_ACCESS_KEY_ID or "").strip()
        secret_key = (settings.AWS_SECRET_ACCESS_KEY or "").strip()

        session_kwargs = {"region_name": settings.AWS_REGION}
        
        if access_key and secret_key:
            # Se as chaves estiverem presentes (Local Dev ou configuradas explicitamente)
            masked_key = f"{access_key[:4]}...{access_key[-4:]}" if len(access_key) > 8 else "****"
            logger.info(f"Usando credenciais estáticas. Key ID: {masked_key}")
            session_kwargs["aws_access_key_id"] = access_key
            session_kwargs["aws_secret_access_key"] = secret_key
        else:
            logger.info(f"Credenciais estáticas não detectadas. Ambiente Lambda: {is_lambda}. Usando IAM Role.")

        session = boto3.Session(**session_kwargs)
        logger.info(f"Sessão AWS inicializada. Provider: {session.get_credentials().method if session.get_credentials() else 'None'}")
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