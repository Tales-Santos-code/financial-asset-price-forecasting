import pytest
import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.api.routers.prediction_router import router 

app = FastAPI()
app.include_router(router)
client = TestClient(app)

@pytest.fixture
def mock_df():
    df = pd.DataFrame({
        "Date": pd.date_range(start="2026-01-01", periods=5),
        "Close": [100.0, 101.0, 102.0, 103.0, 104.0],
        "Volume": [1000, 1100, 1200, 1300, 1400]
    })
    df.set_index("Date", inplace=True)
    return df

@pytest.fixture
def mock_resultado_predict():
    return {
        "symbol": "RACE",
        "current_price": 104.0,
        "predicted_price_tomorrow": 105.5,
        "variation_pct": 1.44,
        "timestamp": "2026-01-05"
    }

PATCH_BASE = "app.api.routers.prediction_router"

def test_predict_stock_ticker_invalido():
    response = client.get("/stock-data-prediction?symbol=FALSO123")
    assert response.status_code == 400
    assert "não suportado" in response.json()["detail"]

@patch(f"{PATCH_BASE}.pipe_to_predict")
@patch(f"{PATCH_BASE}.write_csv_to_s3")
@patch(f"{PATCH_BASE}.read_csv_from_s3")
@patch(f"{PATCH_BASE}.FinanceService")
def test_predict_stock_sucesso(mock_finance_class, mock_read_s3, mock_write_s3, mock_pipe, mock_df, mock_resultado_predict):
    mock_instancia = MagicMock()
    mock_instancia.get_historical_data.return_value = mock_df
    mock_instancia.get_macro_data.return_value = mock_df
    mock_finance_class.return_value = mock_instancia
    mock_read_s3.return_value = pd.DataFrame()
    mock_pipe.return_value = mock_resultado_predict 
    
    response = client.get("/stock-data-prediction?symbol=RACE")
    assert response.status_code == 200
    assert response.json()["symbol"] == "RACE"

@patch(f"{PATCH_BASE}.pipe_to_predict")
@patch(f"{PATCH_BASE}.write_csv_to_s3")
@patch(f"{PATCH_BASE}.read_csv_from_s3")
@patch(f"{PATCH_BASE}.FinanceService")
def test_predict_stock_ticker_minusculo(mock_finance_class, mock_read_s3, mock_write_s3, mock_pipe, mock_df, mock_resultado_predict):
    mock_finance_class.return_value.get_historical_data.return_value = mock_df
    mock_finance_class.return_value.get_macro_data.return_value = mock_df
    mock_read_s3.return_value = pd.DataFrame()
    mock_res = mock_resultado_predict.copy()
    mock_res["symbol"] = "AAPL"
    mock_pipe.return_value = mock_res
    
    response = client.get("/stock-data-prediction?symbol=aapl")
    assert response.status_code == 200
    assert response.json()["symbol"] == "AAPL"

@patch(f"{PATCH_BASE}.read_csv_from_s3")
@patch(f"{PATCH_BASE}.FinanceService")
def test_predict_stock_sem_dados(mock_finance_class, mock_read_s3):
    mock_finance_class.return_value.get_historical_data.return_value = pd.DataFrame()
    mock_read_s3.return_value = pd.DataFrame()
    response = client.get("/stock-data-prediction?symbol=RACE")
    assert response.status_code == 400
    assert "Sem dados para RACE" in response.json()["detail"]

@patch(f"{PATCH_BASE}.FinanceService")
def test_predict_stock_erro_interno(mock_finance_class):
    mock_finance_class.side_effect = Exception("Erro de rede")
    response = client.get("/stock-data-prediction?symbol=RACE")
    assert response.status_code == 500
    assert "Erro interno" in response.json()["detail"]