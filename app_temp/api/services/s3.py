import json
import io
import pandas as pd
from app.api.core.logger import setup_logger

# Importamos o client do nosso novo core centralizado!
from app.api.core.aws import get_s3_client

logger = setup_logger("s3_service")

# Inicializamos o client aqui fora (Singleton) para reaproveitamento
s3_client = get_s3_client()

def read_json_from_s3(bucket: str, key: str) -> dict:
    logger.info(f"Attempting to read JSON from s3://{bucket}/{key}")
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        logger.info(f"Successfully read JSON from s3://{bucket}/{key}")
        return data
    except s3_client.exceptions.NoSuchKey:
        logger.warning(f"Object not found: s3://{bucket}/{key}")
        return None
    except Exception as e:
        logger.error(f"Error reading JSON from s3://{bucket}/{key}: {e}", exc_info=True)
        raise

def write_json_to_s3(bucket: str, key: str, data: dict) -> None:
    logger.info(f"Writing JSON to s3://{bucket}/{key}")
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        )
        logger.info(f"Successfully wrote JSON to s3://{bucket}/{key}")
    except Exception as e:
        logger.error(f"Error writing JSON to s3://{bucket}/{key}: {e}", exc_info=True)
        raise

def read_csv_from_s3(bucket: str, key: str) -> pd.DataFrame:
    logger.info(f"Attempting to read CSV from s3://{bucket}/{key}")
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))
        logger.info(f"Successfully read CSV from s3://{bucket}/{key} (rows: {len(df)})")
        return df
    except s3_client.exceptions.NoSuchKey:
        logger.warning(f"CSV not found: s3://{bucket}/{key}")
        return None
    except Exception as e:
        logger.error(f"Error reading CSV from s3://{bucket}/{key}: {e}", exc_info=True)
        raise

def write_csv_to_s3(bucket: str, key: str, df: pd.DataFrame, mode: str = "w") -> None:
    logger.info(f"Writing CSV to s3://{bucket}/{key} (mode={mode})")
    try:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        s3_client.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue().encode("utf-8"))
        logger.info(f"Successfully wrote CSV to s3://{bucket}/{key} (rows: {len(df)})")
    except Exception as e:
        logger.error(f"Error writing CSV to s3://{bucket}/{key}: {e}", exc_info=True)
        raise

def write_html_to_s3(bucket: str, key: str, html_content: str) -> None:
    logger.info(f"Writing HTML to s3://{bucket}/{key}")
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=html_content.encode("utf-8"),
            ContentType="text/html" # Fundamental para o navegador renderizar
        )
        logger.info(f"Successfully wrote HTML to s3://{bucket}/{key}")
    except Exception as e:
        logger.error(f"Error writing HTML to s3://{bucket}/{key}: {e}", exc_info=True)
        raise

def read_html_from_s3(bucket: str, key: str) -> str:
    logger.info(f"Attempting to read HTML from s3://{bucket}/{key}")
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        html_data = obj["Body"].read().decode("utf-8")
        logger.info(f"Successfully read HTML from s3://{bucket}/{key}")
        return html_data
    except s3_client.exceptions.NoSuchKey:
        logger.warning(f"HTML not found: s3://{bucket}/{key}")
        return None
    except Exception as e:
        logger.error(f"Error reading HTML from s3://{bucket}/{key}: {e}", exc_info=True)
        raise