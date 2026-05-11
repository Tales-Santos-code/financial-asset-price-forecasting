from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch


from app.api.routers.monitoring_router import router 

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

# Teste: barra tickers falsos ou listas
def test_trigger_drift_ticker_invalido():
    response = client.post("/trigger-drift-check?symbol=RACE,AAPL")
    assert response.status_code == 400
    assert "Entrada inválida" in response.json()["detail"]

    # Tenta mandar um ticker inexistente
    response2 = client.post("/trigger-drift-check?symbol=CRIPTOMOEDA")
    assert response2.status_code == 400



@patch("app.api.routers.monitoring_router.check_data_drift")
def test_trigger_drift_individual_sucesso(mock_drift):
    # Passa o RACE (minúsculo para testar o .upper())
    response = client.post("/trigger-drift-check?symbol=race")
    
    assert response.status_code == 200
    dados = response.json()
    
    assert dados["status"] == "Concluído"
    assert "RACE" in dados["message"]
    
    # O TestClient do FastAPI executa as BackgroundTasks no final da requisição.
    mock_drift.assert_called_once_with("RACE")


# Teste:  ativos Em lote
@patch("app.api.routers.monitoring_router.check_data_drift")
def test_trigger_drift_lote_sucesso(mock_drift):
    response = client.post("/trigger-drift-check?symbol=ALL")
    
    assert response.status_code == 200
    dados = response.json()
    
    assert dados["status"] == "Processando Lote"
    
    assert mock_drift.call_count > 1


# ==========================================
# 3. TESTES DA ROTA /drift-report/{symbol}
# ==========================================

def test_view_drift_report_ticker_invalido():
    response = client.get("/drift-report/ACAO_FALSA")
    
    assert response.status_code == 400
    assert "Ticker 'ACAO_FALSA' inválido" in response.json()["detail"]


# Teste: Ticker válido, mas o HTML não existe no S3
@patch("app.api.routers.monitoring_router.read_html_from_s3")
def test_view_drift_report_nao_encontrado(mock_read_s3):
    mock_read_s3.return_value = None
    
    response = client.get("/drift-report/RACE")
    
    assert response.status_code == 404
    assert "Relatório não encontrado no Data Lake" in response.json()["detail"]


# Teste:HTML existe no S3 e é retornado com sucesso
@patch("app.api.routers.monitoring_router.read_html_from_s3")
def test_view_drift_report_sucesso(mock_read_s3):
    mock_html = "<html><body><h1>Drift Report Simulado</h1></body></html>"
    mock_read_s3.return_value = mock_html
    
    response = client.get("/drift-report/RACE")
    
    assert response.status_code == 200
    assert "Drift Report Simulado" in response.text
    