from unittest.mock import patch
import pandas as pd

# ==========================================
# TESTE 1: HEALTH CHECK (Verifica se a API ligou)
# ==========================================
def test_health_check(client):
    response = client.get("/health")
    assert response.status_code in [200, 404]

# ==========================================
# TESTE 2: ENDPOINT DE PREDIÇÃO COM MOCK MLOPS
# ==========================================
# Precisamos "mockar" na ordem inversa em que eles aparecem nos argumentos da função
@patch("app.api.routers.prediction_router.pipe_to_predict")
@patch("app.api.routers.prediction_router.FinanceService.get_macro_data")
@patch("app.api.routers.prediction_router.FinanceService.get_historical_data")
@patch("app.api.routers.prediction_router.write_csv_to_s3")
@patch("app.api.routers.prediction_router.read_csv_from_s3")
def test_predict_stock_success(
    mock_read_s3, 
    mock_write_s3, 
    mock_get_historical, 
    mock_get_macro, 
    mock_predict, 
    client
):
    # 1. PREPARANDO A MENTIRA (O MOCK)
    
    # Fingimos que o S3 está vazio para este teste
    mock_read_s3.return_value = pd.DataFrame()
    
    # Fingimos que o write_csv não faz nada (não salva de verdade)
    mock_write_s3.return_value = None
    
    # Fingimos que o Yahoo Finance retornou 2 dias de dados falsos
    mock_get_historical.return_value = pd.DataFrame(
        {"Close": [390.0, 400.0]},
        index=pd.to_datetime(["2026-05-01", "2026-05-02"])
    )
    
    # Fingimos que a busca macroeconômica retornou vazia
    mock_get_macro.return_value = pd.DataFrame()
    
    # Fingimos que o XGBoost processou os dados com sucesso
    mock_predict.return_value = {
        "symbol": "RACE",
        "current_price": 400.0,
        "predicted_price_tomorrow": 410.0,
        "variation_pct": 2.5,
        "timestamp": "2026-05-02"
    }

    # 2. A AÇÃO (Chama a API como o usuário faria)
    response = client.get("/prod/stock-data-prediction?symbol=RACE")

    # 3. AS VALIDAÇÕES
    assert response.status_code == 200
    
    dados = response.json()
    assert dados["symbol"] == "RACE"
    assert "predicted_price_tomorrow" in dados
    assert dados["variation_pct"] == 2.5

    # Verifica se os métodos do FinanceService foram acionados corretamente pela API
    mock_get_historical.assert_called_once()
    mock_get_macro.assert_called_once()
    mock_predict.assert_called_once()