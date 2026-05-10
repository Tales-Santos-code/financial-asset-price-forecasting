import pandas as pd
import boto3
import io
import os
import logging

# O nome gigante do seu bucket
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "financial-asset-price-forecasting-495599733085-us-east-1-an")

def load_clean_data_from_s3(symbol: str) -> pd.DataFrame:
    """Baixa os dados limpos (com features) direto do S3 para a memória."""
    s3_client = boto3.client('s3')
    key = f"data/processed/{symbol}_historical_cleaned.csv"

    
    print(f"Baixando dados limpos: s3://{BUCKET_NAME}/{key}")
    response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
    csv_content = response['Body'].read().decode('utf-8')
    
    df = pd.read_csv(io.StringIO(csv_content))
    # Garante que a coluna de data seja índice (se existir)
    if 'Date' in df.columns:
        df.set_index('Date', inplace=True)
    return df


def upload_champion_to_s3(local_file_path: str, s3_key: str):
    """Sobe um arquivo para o S3 usando o caminho EXATO fornecido no s3_key."""
    # Instancia o client (ou use o get_s3_client que criamos no core)
    s3_client = boto3.client('s3') 
    bucket_name = "financial-asset-price-forecasting-495599733085-us-east-1-an"

    try:
        # Passa o s3_key purinho, sem concatenar pastas adicionais!
        s3_client.upload_file(local_file_path, bucket_name, s3_key)
        logging.info(f"✅ Arquivo {s3_key} salvo com sucesso no S3.")
    except Exception as e:
        logging.error(f"❌ Erro ao subir {s3_key} para o S3: {e}")


def delete_from_s3(bucket_name: str, key: str):
    """Deleta um arquivo do S3 (usado para limpar scalers obsoletos)."""
    s3_client = boto3.client('s3') # Usa a mesma regra de credenciais que arrumamos
    try:
        s3_client.delete_object(Bucket=bucket_name, Key=key)
        logging.info(f"🗑️ Arquivo obsoleto apagado do S3: {key}")
    except Exception as e:
        logging.warning(f"Não foi possível apagar {key} (pode já não existir). Erro: {e}")