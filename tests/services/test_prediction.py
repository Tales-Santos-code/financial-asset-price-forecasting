import pytest
import pandas as pd
import numpy as np
import io
from unittest.mock import patch, MagicMock

# Import do serviço atualizado
from app.api.services.prediction import (
    _load_from_s3_to_memory,
    _smart_load,
    get_model_and_params,
    pipe_to_predict,
    _model_cache
)

# ==========================================
# FIXTURES GLOBAIS E DADOS FALSOS
# ==========================================

@pytest.fixture(autouse=True)
def clear_model_cache():
    _model_cache.clear()
    yield

@pytest.fixture
def mock_df_history():
    df = pd.DataFrame({"Close": [100.0, 102.0, 105.0]})
    df.index = pd.DatetimeIndex(["2026-01-01", "2026-01-02", "2026-01-03"])
    return df

@pytest.fixture
def mock_df_macro():
    return pd.DataFrame({"SP500_Return": [0.01]})

@pytest.fixture
def mock_df_limpo():
    df = pd.DataFrame({
        "Date": ["2026-01-03"],
        "Target_Log_Return": [0.0],
        "Log_Return": [0.01],
        "RSI_14": [55.0]
    })
    df.set_index("Date", inplace=True)
    return df

PATCH_BASE = "app.api.services.prediction"

# ==========================================
# 1. TESTES DE CARREGAMENTO (S3 -> MEMÓRIA)
# ==========================================

@patch(f"{PATCH_BASE}.get_s3_client")
def test_load_from_s3_to_memory_sucesso(mock_get_s3):
    mock_s3 = MagicMock()
    mock_response = {'Body': MagicMock()}
    mock_response['Body'].read.return_value = b"fake_binary_data"
    mock_s3.get_object.return_value = mock_response
    mock_get_s3.return_value = mock_s3
    
    result = _load_from_s3_to_memory("path/to/artifact.pkl")
    
    assert result == b"fake_binary_data"
    mock_s3.get_object.assert_called_once()

@patch(f"{PATCH_BASE}.joblib.load")
def test_smart_load_joblib(mock_joblib):
    mock_joblib.return_value = "Objeto_Carregado"
    # Um buffer que não começa com PK (não é zip/torch)
    result = _smart_load(b"not_a_zip_file")
    assert result == "Objeto_Carregado"

@patch(f"{PATCH_BASE}.torch.load")
def test_smart_load_torch(mock_torch):
    mock_torch.return_value = "Modelo_Torch"
    # Simula cabeçalho de arquivo ZIP (PyTorch usa internamente)
    result = _smart_load(b"PK\x03\x04_data")
    assert result == "Modelo_Torch"

# ==========================================
# 2. TESTES DE CACHE E PARALELISMO
# ==========================================

@patch(f"{PATCH_BASE}._smart_load")
@patch(f"{PATCH_BASE}._load_from_s3_to_memory")
def test_get_model_and_params_orchestration(mock_load_s3, mock_smart):
    mock_load_s3.return_value = b"raw_data"
    mock_smart.side_effect = ["Pipeline", "Model", "Scaler"]
    
    p, m, s = get_model_and_params("RACE")
    
    assert p == "Pipeline"
    assert m == "Model"
    assert s == "Scaler"
    assert mock_load_s3.call_count == 3 # Pipeline, Model, Scaler
    assert _model_cache["RACE_model"] == "Model"

def test_get_model_and_params_cache_hit():
    _model_cache["NVDA_pipeline"] = "Cached_P"
    _model_cache["NVDA_model"] = "Cached_M"
    _model_cache["NVDA_scaler"] = "Cached_S"
    
    with patch(f"{PATCH_BASE}._load_from_s3_to_memory") as mock_load:
        p, m, s = get_model_and_params("NVDA")
        assert p == "Cached_P"
        mock_load.assert_not_called()

# ==========================================
# 3. TESTES DE INFERÊNCIA (pipe_to_predict)
# ==========================================

@patch(f"{PATCH_BASE}.save_prediction_log")
@patch(f"{PATCH_BASE}.get_model_and_params")
def test_pipe_to_predict_fluxo_completo(mock_get_params, mock_save_log, mock_df_history, mock_df_macro, mock_df_limpo):
    # Setup Mocks de Pipeline e Modelo
    mock_pipeline = MagicMock()
    mock_pipeline.transform.return_value = mock_df_limpo
    
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.02]
    mock_model.n_features_in_ = 26
    # Remove atributos que o MagicMock cria automaticamente e que quebram o código
    del mock_model.num_features
    del mock_model.num_feature
    
    mock_scaler = MagicMock()
    mock_scaler.n_features_in_ = 27
    mock_scaler.transform.return_value = np.zeros((1, 27))
    mock_scaler.inverse_transform.return_value = np.zeros((1, 27))
    
    mock_get_params.return_value = (mock_pipeline, mock_model, mock_scaler)
    
    resultado = pipe_to_predict("RACE", mock_df_history, mock_df_macro)
    
    assert resultado["symbol"] == "RACE"
    assert "predicted_price_tomorrow" in resultado
    assert resultado["current_price"] == 105.0
    mock_save_log.assert_called_once()