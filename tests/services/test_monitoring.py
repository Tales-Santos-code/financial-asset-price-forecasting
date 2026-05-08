import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime

# Ajuste o caminho se o seu arquivo estiver em outro lugar
from app.api.services.monitoring import save_prediction_log

# ==========================================
# FIXTURES E DADOS FALSOS (MOCKS)
# ==========================================
@pytest.fixture
def mock_features():
    """Simula a exata linha de dados (features) que foi passada para o modelo."""
    return pd.DataFrame({
        "RSI_14": [55.5],
        "Volume": [10500],
        "SP500_Return": [0.012]
    })

@pytest.fixture
def mock_prediction():
    """Simula o dicionário de resposta gerado pelo modelo."""
    return {
        "symbol": "RACE",
        "predicted_price_tomorrow": 105.5,
        "variation_pct": 2.5
    }

# Caminho base para injetar os Mocks
PATCH_BASE = "app.api.services.monitoring"


# ==========================================
# 1. TESTE DO CAMINHO FELIZ (Tudo Salvo Corretamente)
# ==========================================
@patch(f"{PATCH_BASE}.write_json_to_s3")
@patch(f"{PATCH_BASE}.os.makedirs")
@patch("builtins.open", new_callable=mock_open)
@patch(f"{PATCH_BASE}.datetime") # Mockamos o relógio para a data não mudar em cada teste
def test_save_prediction_log_sucesso(mock_datetime, mock_file_open, mock_makedirs, mock_write_s3, mock_features, mock_prediction):
    
    # 1. Congelamos o tempo em uma data exata
    mock_now = datetime(2026, 5, 7, 12, 30, 0, 0)
    mock_datetime.utcnow.return_value = mock_now
    
    # AÇÃO
    save_prediction_log("RACE", mock_features, mock_prediction)
    
    # VERIFICAÇÕES DO S3:
    mock_write_s3.assert_called_once()
    
    # Extraímos os argumentos que a função de S3 recebeu
    args, kwargs = mock_write_s3.call_args
    s3_key_enviada = args[1]
    payload_enviado = args[2]
    
    # O arquivo tem o nome com a data congelada correta?
    assert s3_key_enviada == "predictions/RACE/2026-05-07T12-30-00-000000.json"
    
    # O Payload tem tudo que o Evidently AI precisa?
    assert payload_enviado["symbol"] == "RACE"
    assert payload_enviado["model_version"] == "champion"
    assert payload_enviado["features_input"] == {"RSI_14": 55.5, "Volume": 10500, "SP500_Return": 0.012}
    assert payload_enviado["prediction_output"] == mock_prediction
    
    # VERIFICAÇÕES DO ARQUIVO LOCAL:
    mock_makedirs.assert_called_once()
    mock_file_open.assert_called_once()


# ==========================================
# 2. TESTE DE RESILIÊNCIA DE DADOS (Features Vazias)
# ==========================================
@patch(f"{PATCH_BASE}.write_json_to_s3")
@patch(f"{PATCH_BASE}.os.makedirs")
@patch("builtins.open", new_callable=mock_open)
def test_save_prediction_log_features_vazias(mock_file_open, mock_makedirs, mock_write_s3, mock_prediction):
    # Se, por um bug bizarro, as features vierem vazias, o código não pode quebrar o `.to_dict()[0]`
    df_vazio = pd.DataFrame()
    
    # AÇÃO
    save_prediction_log("RACE", df_vazio, mock_prediction)
    
    # VERIFICAÇÃO
    args, kwargs = mock_write_s3.call_args
    payload_enviado = args[2]
    
    # Ele tem que ter injetado um dicionário vazio {} para não dar KeyError depois
    assert payload_enviado["features_input"] == {}


# ==========================================
# 3. TESTE DE FAIL-SAFE DO S3 (Nuvem Caiu)
# ==========================================
@patch(f"{PATCH_BASE}.write_json_to_s3")
@patch(f"{PATCH_BASE}.os.makedirs")
@patch("builtins.open", new_callable=mock_open)
def test_save_prediction_log_falha_s3(mock_file_open, mock_makedirs, mock_write_s3, mock_features, mock_prediction):
    # Forçamos um erro gravíssimo de conexão na AWS
    mock_write_s3.side_effect = Exception("AWS Access Denied ou Bucket Not Found")
    
    # AÇÃO:
    try:
        save_prediction_log("RACE", mock_features, mock_prediction)
    except Exception as e:
        # Se entrar aqui, o teste FALHA, porque a API do usuário seria travada com Erro 500
        pytest.fail(f"A função deveria ter engolido o erro do S3 silenciosamente, mas levantou: {e}")
        
    # Verificamos que, como o S3 explodiu na primeira linha do Try, ele NEM TENTOU salvar localmente
    mock_makedirs.assert_not_called()


# ==========================================
# 4. TESTE DE FAIL-SAFE DO DISCO (Erro de Permissão)
# ==========================================
@patch(f"{PATCH_BASE}.write_json_to_s3")
@patch(f"{PATCH_BASE}.os.makedirs")
@patch("builtins.open") # Sem mock_open aqui, para podermos forçar um Erro de I/O
def test_save_prediction_log_falha_disco_local(mock_file_open, mock_makedirs, mock_write_s3, mock_features, mock_prediction):
    # S3 funcionou, mas salvar no disco local deu erro de permissão (disco cheio ou readonly)
    mock_file_open.side_effect = PermissionError("Acesso negado na pasta /data")
    
    # AÇÃO:
    try:
        save_prediction_log("RACE", mock_features, mock_prediction)
    except Exception as e:
        pytest.fail(f"A função deveria ter engolido o erro do disco local, mas levantou: {e}")
        
    # VERIFICAÇÃO MÁGICA:
    # Mesmo dando pau no disco, como o S3 roda ANTES no código, o S3 DEVE ter sido salvo!
    mock_write_s3.assert_called_once()