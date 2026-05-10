import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

# Ajuste este import para o local exato do seu arquivo cleaning.py
from app.api.services.cleaning import historical_cleaning

# ==========================================
# FIXTURES E DADOS FALSOS (MOCKS)
# ==========================================
@pytest.fixture
def mock_df_history():
    """Gera um histórico cru falso (antes da limpeza)."""
    return pd.DataFrame({
        "Date": ["2026-01-01", "2026-01-02"],
        "Close": [100.0, 105.0]
    })

@pytest.fixture
def mock_df_macro():
    """Gera dados macroeconômicos falsos."""
    return pd.DataFrame({
        "Date": ["2026-01-01", "2026-01-02"],
        "SP500_Return": [0.01, -0.01]
    })

@pytest.fixture
def mock_df_cleaned():
    """Simula o DataFrame perfeito e limpo que sai do seu pipeline."""
    return pd.DataFrame({
        "Date": ["2026-01-01", "2026-01-02"],
        "Close": [100.0, 105.0],
        "SP500_Return": [0.01, -0.01],
        "RSI_14": [50.0, 55.0]
    })

@pytest.fixture
def mock_pipeline():
    """Simula a classe FeatureEngineering (O 'pipeline_ferrari')."""
    pipeline = MagicMock()
    # is_training começa como True para testar se a função muda para False
    pipeline.is_training = True 
    return pipeline


# ==========================================
# TESTES DO SERVIÇO DE LIMPEZA
# ==========================================

# Base para os Mocks não ficarem gigantes
PATCH_BASE = "app.api.services.cleaning"


# 1. TESTE DE SEGURANÇA (Early Exit)
# Se o arquivo não existir no S3, o script não pode tentar processar nada!
@patch(f"{PATCH_BASE}.read_csv_from_s3")
@patch(f"{PATCH_BASE}.get_model_and_params")
@patch(f"{PATCH_BASE}.write_csv_to_s3")
def test_historical_cleaning_sem_dados_no_s3(mock_write_s3, mock_get_model, mock_read_s3):
    # Setup: Finge que o S3 retornou None (Arquivo não existe)
    mock_read_s3.return_value = None
    
    # Ação
    historical_cleaning("RACE")
    
    # Verificação:
    # 1. Ele DEVE ter tentado ler do S3
    mock_read_s3.assert_called_once()
    
    # 2. Ele NÃO DEVE ter tentado carregar o modelo de ML da RAM (parou no if)
    mock_get_model.assert_not_called()
    
    # 3. Ele NÃO DEVE ter tentado salvar nada de volta na AWS
    mock_write_s3.assert_not_called()


# 2. TESTE DO CAMINHO FELIZ (Transformação Simples, sem DF Macro)
@patch(f"{PATCH_BASE}.read_csv_from_s3")
@patch(f"{PATCH_BASE}.get_model_and_params")
@patch(f"{PATCH_BASE}.write_csv_to_s3")
def test_historical_cleaning_sucesso_simples(mock_write_s3, mock_get_model, mock_read_s3, mock_df_history, mock_df_cleaned, mock_pipeline):
    # Setup
    mock_read_s3.return_value = mock_df_history
    mock_pipeline.transform.return_value = mock_df_cleaned
    # A função get_model_and_params retorna uma tupla (pipeline, model), mockamos os dois
    mock_get_model.return_value = (mock_pipeline, MagicMock(), None)
    
    # Ação (Chamando sem o df_macro)
    historical_cleaning("RACE")
    
    # Verificações:
    # 1. O pipeline DEVE ter sido ajustado para modo de Inferência (is_training = False)
    assert mock_pipeline.is_training is False
    
    # 2. A transformação deve ter rodado passando APENAS o df_history
    mock_pipeline.transform.assert_called_once_with(mock_df_history)
    
    # 3. O CSV final deve ter sido salvo no bucket Processed
    mock_write_s3.assert_called_once()
    args, kwargs = mock_write_s3.call_args
    assert kwargs["key"] == "data/processed/RACE_historical_cleaned.csv"


# 3. TESTE DO CAMINHO FELIZ (Transformação Tupla, com DF Macro)
@patch(f"{PATCH_BASE}.read_csv_from_s3")
@patch(f"{PATCH_BASE}.get_model_and_params")
@patch(f"{PATCH_BASE}.write_csv_to_s3")
def test_historical_cleaning_sucesso_com_macro(mock_write_s3, mock_get_model, mock_read_s3, mock_df_history, mock_df_macro, mock_df_cleaned, mock_pipeline):
    # Setup
    mock_read_s3.return_value = mock_df_history
    mock_pipeline.transform.return_value = mock_df_cleaned
    mock_get_model.return_value = (mock_pipeline, MagicMock(), None)
    
    # Ação (Passando o df_macro)
    historical_cleaning("RACE", df_macro=mock_df_macro)
    
    # Verificação: A transformação tem que ser chamada passando a Tupla (history, macro)
    mock_pipeline.transform.assert_called_once_with((mock_df_history, mock_df_macro))
    mock_write_s3.assert_called_once()


# 4. TESTE DE RESILIÊNCIA (Fallback Funciona?)
@patch(f"{PATCH_BASE}.read_csv_from_s3")
@patch(f"{PATCH_BASE}.get_model_and_params")
@patch(f"{PATCH_BASE}.write_csv_to_s3")
def test_historical_cleaning_fallback_exception(mock_write_s3, mock_get_model, mock_read_s3, mock_df_history, mock_df_macro, mock_df_cleaned, mock_pipeline):
    # Setup
    mock_read_s3.return_value = mock_df_history
    mock_get_model.return_value = (mock_pipeline, MagicMock(), None)
    
    # Finge que o "try" de rodar com a tupla vai dar ERRO, e a segunda tentativa vai dar CERTO
    mock_pipeline.transform.side_effect = [
        Exception("Erro de formato de tupla"), # 1ª chamada (Try) falha
        mock_df_cleaned                        # 2ª chamada (Except Fallback) dá certo
    ]
    
    # Ação
    historical_cleaning("RACE", df_macro=mock_df_macro)
    
    # Verificação: O Fallback salvou o dia?
    # O pipeline deve ter sido chamado exatas DUAS vezes!
    assert mock_pipeline.transform.call_count == 2
    # E mesmo com o erro na primeira tentativa, o arquivo tem que ser salvo no final
    mock_write_s3.assert_called_once()