import pytest
import pandas as pd
import json
from unittest.mock import patch, MagicMock

# Ajuste o caminho se necessário
from app.api.services.s3 import (
    read_json_from_s3, write_json_to_s3,
    read_csv_from_s3, write_csv_to_s3,
    read_html_from_s3, write_html_to_s3
)

# ==========================================
# FIXTURE DO S3 (O "Distorcedor de Realidade")
# ==========================================
@pytest.fixture
def mock_s3_client():
    """
    Mock global para o s3_client.
    Como o seu s3_client é instanciado no topo do arquivo s3.py (Singleton), 
    nós precisamos interceptá-lo lá dentro.
    """
    with patch("app.api.services.s3.s3_client") as mock:
        # Criamos uma exceção falsa para simular o comportamento do Boto3
        class FakeNoSuchKey(Exception):
            pass
        
        # Injetamos essa exceção falsa dentro do mock para o bloco 'except' funcionar
        mock.exceptions.NoSuchKey = FakeNoSuchKey
        yield mock


# ==========================================
# 1. TESTES DE JSON
# ==========================================
def test_read_json_from_s3_sucesso(mock_s3_client):
    # Setup: Simulando o retorno binário da AWS
    dado_falso = {"status": "ok", "valor": 42}
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps(dado_falso).encode("utf-8")
    mock_s3_client.get_object.return_value = {"Body": mock_body}
    
    # Ação
    resultado = read_json_from_s3("meu-bucket", "pasta/arquivo.json")
    
    # Verificação
    assert resultado == dado_falso
    mock_s3_client.get_object.assert_called_once_with(Bucket="meu-bucket", Key="pasta/arquivo.json")

def test_read_json_from_s3_not_found(mock_s3_client):
    # Setup: Simulando arquivo inexistente (Gatilho da exceção NoSuchKey)
    mock_s3_client.get_object.side_effect = mock_s3_client.exceptions.NoSuchKey()
    
    # Ação
    resultado = read_json_from_s3("meu-bucket", "fantasma.json")
    
    # Verificação: Seu código prometeu retornar None quando não acha
    assert resultado is None

def test_write_json_to_s3_sucesso(mock_s3_client):
    dados = {"nome": "Ferrari"}
    
    # Ação
    write_json_to_s3("meu-bucket", "teste.json", dados)
    
    # Verificação
    mock_s3_client.put_object.assert_called_once()
    args, kwargs = mock_s3_client.put_object.call_args
    assert kwargs["Bucket"] == "meu-bucket"
    assert kwargs["Key"] == "teste.json"
    assert b'"nome": "Ferrari"' in kwargs["Body"]


# ==========================================
# 2. TESTES DE CSV (O mais crítico para ML)
# ==========================================
def test_read_csv_from_s3_sucesso(mock_s3_client):
    # Setup: Simulando um CSV em bytes vindo da rede
    csv_bytes = b"Date,Close\n2026-01-01,100.5\n2026-01-02,102.0"
    mock_body = MagicMock()
    mock_body.read.return_value = csv_bytes
    mock_s3_client.get_object.return_value = {"Body": mock_body}
    
    # Ação
    df = read_csv_from_s3("bucket-ml", "dados.csv")
    
    # Verificação: O pandas conseguiu reconstruir o DataFrame?
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert df["Close"].iloc[0] == 100.5

def test_read_csv_from_s3_not_found(mock_s3_client):
    # Setup: Exceção nativa da AWS
    mock_s3_client.get_object.side_effect = mock_s3_client.exceptions.NoSuchKey()
    
    # Ação
    df = read_csv_from_s3("bucket-ml", "historico_falso.csv")
    
    # Verificação
    assert df is None

def test_write_csv_to_s3_sucesso(mock_s3_client):
    # Criamos um DataFrame de mentira
    df_falso = pd.DataFrame({"Preco": [10, 20]})
    
    # Ação
    write_csv_to_s3("bucket", "novo.csv", df_falso)
    
    # Verificação
    mock_s3_client.put_object.assert_called_once()
    kwargs = mock_s3_client.put_object.call_args[1]
    
    # Garantimos que os dados corretos foram convertidos e anexados no envio
    assert b"Preco" in kwargs["Body"]
    assert b"10" in kwargs["Body"]


# ==========================================
# 3. TESTES DE HTML (Para o Evidently AI)
# ==========================================
def test_write_html_to_s3_sucesso(mock_s3_client):
    html_falso = "<html><body>Drift</body></html>"
    
    # Ação
    write_html_to_s3("bucket", "dashboard.html", html_falso)
    
    # Verificação
    mock_s3_client.put_object.assert_called_once()
    kwargs = mock_s3_client.put_object.call_args[1]
    
    # REGRA DE OURO DO S3: Se não tiver ContentType="text/html", o navegador faz 
    # download do arquivo em vez de exibir a página. Vamos garantir que seu código enviou isso:
    assert kwargs["ContentType"] == "text/html"
    assert kwargs["Body"] == html_falso.encode("utf-8")

def test_read_html_from_s3_sucesso(mock_s3_client):
    mock_body = MagicMock()
    mock_body.read.return_value = b"<h1>Relatorio</h1>"
    mock_s3_client.get_object.return_value = {"Body": mock_body}
    
    html = read_html_from_s3("bucket", "report.html")
    
    assert html == "<h1>Relatorio</h1>"

def test_read_html_from_s3_erro_critico(mock_s3_client):
    # Se der um erro diferente de NoSuchKey (ex: AccessDenied), o sistema DEVE quebrar (raise)
    mock_s3_client.get_object.side_effect = Exception("AWS Access Denied")
    
    with pytest.raises(Exception) as excinfo:
        read_html_from_s3("bucket", "report.html")
        
    assert "AWS Access Denied" in str(excinfo.value)