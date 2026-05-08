import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch

# Importe a rota de monitoramento
# Ajuste o caminho se o nome do seu arquivo ou pasta for diferente
from app.api.routers.monitoring_router import router 
from app.api.schemas.prediction_schema import StockSymbol

# Cria um app FastAPI "falso" apenas para testar essas rotas
app = FastAPI()
app.include_router(router)

client = TestClient(app)

# ==========================================
# 1. TESTES DA ROTA /health
# ==========================================
def test_check_model_health():
    """Garante que o Load Balancer da AWS vai receber o sinal de vida do modelo"""
    response = client.get("/health")
    
    assert response.status_code == 200
    dados = response.json()
    
    # Verifica se os campos obrigatórios estão no JSON
    assert dados["is_online"] is True
    assert "model_version" in dados
    assert "total_predictions_today" in dados


# ==========================================
# 2. TESTES DA ROTA /trigger-drift-check
# ==========================================

# Teste: O Porteiro barra tickers falsos ou listas?
def test_trigger_drift_ticker_invalido():
    # Tenta mandar uma lista separada por vírgula
    response = client.post("/trigger-drift-check?symbol=RACE,AAPL")
    assert response.status_code == 400
    assert "Entrada inválida" in response.json()["detail"]

    # Tenta mandar um ticker inexistente
    response2 = client.post("/trigger-drift-check?symbol=CRIPTOMOEDA")
    assert response2.status_code == 400


# Teste: O Caminho Feliz para UM ativo
# Mockamos o check_data_drift para que o Evidently não rode de verdade durante o teste
@patch("app.api.routers.monitoring_router.check_data_drift")
def test_trigger_drift_individual_sucesso(mock_drift):
    # Passa o RACE (minúsculo para testar o .upper())
    response = client.post("/trigger-drift-check?symbol=race")
    
    assert response.status_code == 200
    dados = response.json()
    
    assert dados["status"] == "Processando Individual"
    assert "RACE" in dados["message"]
    
    # O TestClient do FastAPI executa as BackgroundTasks no final da requisição.
    # Como mockamos, podemos verificar se a função foi "chamada" corretamente nos bastidores!
    mock_drift.assert_called_once_with("RACE")


# Teste: O Caminho Feliz para TODOS os ativos (Em lote)
@patch("app.api.routers.monitoring_router.check_data_drift")
def test_trigger_drift_lote_sucesso(mock_drift):
    response = client.post("/trigger-drift-check?symbol=ALL")
    
    assert response.status_code == 200
    dados = response.json()
    
    assert dados["status"] == "Processando Lote"
    
    # O mock_drift deve ter sido chamado várias vezes (uma para cada ativo, exceto o ^GSPC)
    # Como temos 6 ações e 1 índice macro, ele deve ter sido chamado 6 vezes
    assert mock_drift.call_count > 1


# ==========================================
# 3. TESTES DA ROTA /drift-report/{symbol}
# ==========================================

# Teste: O Porteiro barra tentativa de hackear a URL (Path Traversal ou ticker falso)
def test_view_drift_report_ticker_invalido():
    response = client.get("/drift-report/ACAO_FALSA")
    
    assert response.status_code == 400
    assert "Ticker 'ACAO_FALSA' inválido" in response.json()["detail"]


# Teste: Ticker válido, mas o HTML não existe no S3 (ainda não rodou o drift)
@patch("app.api.routers.monitoring_router.read_html_from_s3")
def test_view_drift_report_nao_encontrado(mock_read_s3):
    # Finge que o S3 não achou o arquivo e retornou None
    mock_read_s3.return_value = None
    
    response = client.get("/drift-report/RACE")
    
    assert response.status_code == 404
    assert "Relatório não encontrado no Data Lake" in response.json()["detail"]


# Teste: Caminho Feliz (HTML existe no S3 e é retornado com sucesso)
@patch("app.api.routers.monitoring_router.read_html_from_s3")
def test_view_drift_report_sucesso(mock_read_s3):
    # Finge que o S3 achou o arquivo e devolveu o HTML
    mock_html = "<html><body><h1>Drift Report Simulado</h1></body></html>"
    mock_read_s3.return_value = mock_html
    
    response = client.get("/drift-report/RACE")
    
    assert response.status_code == 200
    # Como a resposta é da classe HTMLResponse, verificamos o texto puro
    assert "Drift Report Simulado" in response.text
    