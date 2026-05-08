import pytest
import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Importa o router
from app.api.routers.prediction_router import router 

# Monta uma aplicação "falsa" só para rodar os testes
app = FastAPI()
app.include_router(router)

client = TestClient(app)

# ==========================================
# FIXTURES (DADOS FALSOS)
# ==========================================
@pytest.fixture
def mock_df():
    """Gera um DataFrame minúsculo simulando os dados de ações."""
    df = pd.DataFrame({
        "Date": pd.date_range(start="2026-01-01", periods=5),
        "Close": [100.0, 101.0, 102.0, 103.0, 104.0],
        "Volume": [1000, 1100, 1200, 1300, 1400]
    })
    df.set_index("Date", inplace=True)
    return df

@pytest.fixture
def mock_resultado_predict():
    """Simula o dicionário exato que a API espera no Schema PredictionResponse"""
    return {
        "symbol": "RACE",
        "current_price": 104.0,
        "predicted_price_tomorrow": 105.5,
        "variation_pct": 1.44,
        "timestamp": "2026-01-05"
    }

# ==========================================
# TESTES
# ==========================================

# Definindo o caminho base do patch para não repetirmos código
PATCH_BASE = "app.api.routers.prediction_router"


# 1. TESTE DO PORTEIRO (TICKER INVÁLIDO) - Não precisa de Mocks!
def test_predict_stock_ticker_invalido():
    # Ação: Passamos um ticker falso
    response = client.get("/stock-data-prediction?symbol=FALSO123")
    
    # Verificação: Deve ser barrado instantaneamente com status 400
    assert response.status_code == 400
    dados = response.json()
    assert "não é suportado" in dados["detail"]


# 2. TESTE DO CAMINHO FELIZ (SUCESSO)
@patch(f"{PATCH_BASE}.pipe_to_predict")
@patch(f"{PATCH_BASE}.write_csv_to_s3")
@patch(f"{PATCH_BASE}.read_csv_from_s3")
@patch(f"{PATCH_BASE}.FinanceService")
def test_predict_stock_sucesso(mock_finance_class, mock_read_s3, mock_write_s3, mock_pipe, mock_df, mock_resultado_predict):
    # Setup Mocks
    mock_instancia = MagicMock()
    mock_instancia.get_historical_data.return_value = mock_df
    mock_instancia.get_macro_data.return_value = mock_df
    mock_finance_class.return_value = mock_instancia
    
    mock_read_s3.return_value = pd.DataFrame() # Finge que o S3 estava vazio
    mock_pipe.return_value = mock_resultado_predict 
    
    # Ação
    response = client.get("/stock-data-prediction?symbol=RACE")
    
    # Verificação
    assert response.status_code == 200
    # Verifica se os mocks pesados foram chamados
    mock_read_s3.assert_called_once()
    mock_write_s3.assert_called_once()
    mock_pipe.assert_called_once()
    
    # Validações com as chaves corretas
    dados = response.json()
    assert dados["symbol"] == "RACE"
    assert dados["predicted_price_tomorrow"] == 105.5


# 3. TESTE DE RESILIÊNCIA (LETRAS MINÚSCULAS)
@patch(f"{PATCH_BASE}.pipe_to_predict")
@patch(f"{PATCH_BASE}.write_csv_to_s3")
@patch(f"{PATCH_BASE}.read_csv_from_s3")
@patch(f"{PATCH_BASE}.FinanceService")
def test_predict_stock_ticker_minusculo(mock_finance_class, mock_read_s3, mock_write_s3, mock_pipe, mock_df, mock_resultado_predict):
    # Setup Mocks
    mock_finance_class.return_value.get_historical_data.return_value = mock_df
    mock_finance_class.return_value.get_macro_data.return_value = mock_df
    mock_read_s3.return_value = pd.DataFrame()
    
    mock_resultado_aapl = mock_resultado_predict.copy()
    mock_resultado_aapl["symbol"] = "AAPL"
    mock_pipe.return_value = mock_resultado_aapl
    
    # Ação: Passamos "aapl" em letras minúsculas
    response = client.get("/stock-data-prediction?symbol=aapl")
    
    # Verificação: Tem que dar status 200 e ter convertido internamente
    assert response.status_code == 200
    assert response.json()["symbol"] == "AAPL"


# 4. TESTE DE REGRA DE NEGÓCIO (DADOS VAZIOS/FALHA NO YFINANCE)
@patch(f"{PATCH_BASE}.read_csv_from_s3")
@patch(f"{PATCH_BASE}.FinanceService")
def test_predict_stock_sem_dados(mock_finance_class, mock_read_s3):
    # Finge que o Yahoo Finance retornou um DataFrame vazio
    mock_finance_class.return_value.get_historical_data.return_value = pd.DataFrame()
    # Finge que o S3 também retornou vazio
    mock_read_s3.return_value = pd.DataFrame()
    
    # Ação
    response = client.get("/stock-data-prediction?symbol=RACE")
    
    # Verificação: Como deu vazio em ambos, a API tem que estourar o ValueError (Status 400)
    assert response.status_code == 400
    assert "Sem dados encontrados para o ticker RACE" in response.json()["detail"]


# 5. TESTE DE ERRO INESPERADO DO SERVIDOR (HTTP 500)
@patch(f"{PATCH_BASE}.FinanceService")
def test_predict_stock_erro_interno(mock_finance_class):
    # Finge que a biblioteca FinanceService quebrou "do nada" (ex: erro de conexão HTTP interno)
    mock_finance_class.side_effect = Exception("Erro catastrófico de rede")
    
    # Ação
    response = client.get("/stock-data-prediction?symbol=RACE")
    
    # Verificação: O bloco 'except Exception' deve engolir o erro e devolver um 500 elegante
    assert response.status_code == 500
    assert response.json()["detail"] == "Ocorreu um erro ao processar a predição."