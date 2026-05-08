import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Import do serviço. Ajuste o caminho se necessário.
from app.api.services.prediction import (
    _download_model_from_s3,
    get_model_and_params,
    pipe_to_predict,
    _model_cache
)

# ==========================================
# FIXTURES GLOBAIS E DADOS FALSOS
# ==========================================

# Limpa a memória RAM (Cache) antes de CADA teste para não haver interferência
@pytest.fixture(autouse=True)
def clear_model_cache():
    _model_cache.clear()
    yield

@pytest.fixture
def mock_df_history():
    """Histórico base simulado para inferência"""
    df = pd.DataFrame({
        "Close": [100.0, 102.0, 105.0]
    })
    df.index = pd.DatetimeIndex(["2026-01-01", "2026-01-02", "2026-01-03"])
    return df

@pytest.fixture
def mock_df_macro():
    """Macro simulado"""
    return pd.DataFrame({"SP500_Return": [0.01]})

@pytest.fixture
def mock_df_limpo():
    """Simula o DataFrame 100% tratado saindo do Pipeline do Pandas"""
    df = pd.DataFrame({
        "Date": ["2026-01-03"],
        "Target_Log_Return": [np.nan], # A coluna que queremos prever
        "RSI_14": [55.0],
        "Volume": [1000]
    })
    df.set_index("Date", inplace=True)
    return df

# Caminho base para injetar os Mocks
PATCH_BASE = "app.api.services.prediction"

# ==========================================
# 1. TESTES DE DOWNLOAD DO S3
# ==========================================
@patch(f"{PATCH_BASE}.get_s3_client")
def test_download_model_from_s3_sucesso(mock_get_s3):
    mock_s3_client = MagicMock()
    mock_get_s3.return_value = mock_s3_client
    
    _download_model_from_s3("modelo_RACE.pkl", "/tmp/modelo_RACE.pkl")
    
    mock_s3_client.download_file.assert_called_once()
    args = mock_s3_client.download_file.call_args[0]
    assert args[1] == "modelo_RACE.pkl"
    assert args[2] == "/tmp/modelo_RACE.pkl"

@patch(f"{PATCH_BASE}.get_s3_client")
def test_download_model_from_s3_falha(mock_get_s3):
    mock_s3_client = MagicMock()
    mock_s3_client.download_file.side_effect = Exception("S3 Indisponível")
    mock_get_s3.return_value = mock_s3_client
    
    with pytest.raises(Exception) as excinfo:
        _download_model_from_s3("modelo_RACE.pkl", "/tmp/modelo_RACE.pkl")
    assert "S3 Indisponível" in str(excinfo.value)

# ==========================================
# 2. TESTES DE GERENCIAMENTO DE MEMÓRIA (CACHE & JOBLIB)
# ==========================================
@patch(f"{PATCH_BASE}.joblib.load")
@patch(f"{PATCH_BASE}._download_model_from_s3")
def test_get_model_and_params_com_scaler(mock_download, mock_joblib_load):
    # CORREÇÃO 1: A ordem correta aqui!
    mock_joblib_load.side_effect = ["Scaler_Mock", "Pipeline_Mock", "Model_Mock"]
    
    pipeline, model, scaler = get_model_and_params("RACE")
    
    assert pipeline == "Pipeline_Mock"
    assert model == "Model_Mock"
    assert scaler == "Scaler_Mock"
    
    assert _model_cache["RACE_pipeline"] == "Pipeline_Mock"
    assert mock_download.call_count == 3

@patch(f"{PATCH_BASE}.joblib.load")
@patch(f"{PATCH_BASE}._download_model_from_s3")
def test_get_model_and_params_arvore_sem_scaler(mock_download, mock_joblib_load):
    def download_side_effect(s3_key, local_path):
        if "scaler" in s3_key:
            raise Exception("Arquivo não encontrado")
    
    mock_download.side_effect = download_side_effect
    mock_joblib_load.side_effect = ["Pipeline_Mock", "Model_Mock"]
    
    pipeline, model, scaler = get_model_and_params("VALE3")
    
    assert pipeline == "Pipeline_Mock"
    assert model == "Model_Mock"
    assert scaler is None

def test_get_model_and_params_cache_hit():
    _model_cache["AAPL_pipeline"] = "Pipe_Cached"
    _model_cache["AAPL_model"] = "Model_Cached"
    _model_cache["AAPL_scaler"] = "Scaler_Cached"
    
    with patch(f"{PATCH_BASE}._download_model_from_s3") as mock_download:
        pipeline, model, scaler = get_model_and_params("AAPL")
        
        assert pipeline == "Pipe_Cached"
        assert model == "Model_Cached"
        mock_download.assert_not_called()

# ==========================================
# 3. TESTES DA INFERÊNCIA E S3 (pipe_to_predict)
# ==========================================
@patch(f"{PATCH_BASE}.save_prediction_log")
@patch(f"{PATCH_BASE}.write_csv_to_s3")
@patch(f"{PATCH_BASE}.read_csv_from_s3")
@patch(f"{PATCH_BASE}.get_model_and_params")
def test_pipe_to_predict_caminho_feliz_com_scaler(mock_get_model, mock_read_s3, mock_write_s3, mock_save_log, mock_df_history, mock_df_macro, mock_df_limpo):
    mock_pipeline = MagicMock()
    mock_pipeline.transform.return_value = mock_df_limpo
    
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.array([[0.5, 1.5]])
    
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.05]
    
    mock_get_model.return_value = (mock_pipeline, mock_model, mock_scaler)
    mock_read_s3.return_value = pd.DataFrame({"Date": ["2026-01-01"], "RSI_14": [40.0]})
    
    resultado = pipe_to_predict("RACE", mock_df_history, mock_df_macro)
    
    assert resultado["symbol"] == "RACE"
    assert resultado["current_price"] == 105.0 
    assert "predicted_price_tomorrow" in resultado
    assert resultado["timestamp"] == "2026-01-03"
    
    args_scaler = mock_scaler.transform.call_args[0][0]
    assert "Target_Log_Return" not in args_scaler.columns
    mock_write_s3.assert_called_once()
    mock_save_log.assert_called_once()


# CORREÇÃO 2: Apontando o patch direto para o arquivo cleaning.py!
@patch("app.api.services.cleaning.historical_cleaning")
@patch(f"{PATCH_BASE}.save_prediction_log")
@patch(f"{PATCH_BASE}.read_csv_from_s3")
@patch(f"{PATCH_BASE}.get_model_and_params")
def test_pipe_to_predict_aciona_limpeza_historica(mock_get_model, mock_read_s3, mock_save_log, mock_historical_cleaning, mock_df_history, mock_df_macro, mock_df_limpo):
    mock_pipeline = MagicMock()
    mock_pipeline.transform.return_value = mock_df_limpo
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.01]
    
    mock_get_model.return_value = (mock_pipeline, mock_model, None)
    mock_read_s3.return_value = None
    
    pipe_to_predict("RACE", mock_df_history, mock_df_macro)
    
    mock_historical_cleaning.assert_called_once_with("RACE", mock_df_macro)