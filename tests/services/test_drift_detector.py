import pytest
import os
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open

# Ajuste o caminho se o seu arquivo estiver em outro lugar
from app.api.services.drift_detector import check_data_drift, disparar_retreino_github, load_production_logs

# ==========================================
# FIXTURES E DADOS FALSOS (MOCKS)
# ==========================================
@pytest.fixture
def mock_df_referencia():
    """Simula os dados originais usados no treinamento do modelo."""
    return pd.DataFrame({
        "Date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"],
        "Target_Log_Return": [0.01, -0.01, 0.02, 0.01, -0.02],
        "RSI_14": [50, 52, 45, 60, 55],
        "Volume": [1000, 1100, 1200, 1300, 1400]
    })

@pytest.fixture
def mock_df_producao():
    """Simula as predições feitas em tempo real (mais de 5 para o Evidently rodar)."""
    return pd.DataFrame({
        "RSI_14": [51, 53, 44, 61, 56, 50],
        "Volume": [1050, 1150, 1250, 1350, 1450, 1000]
    })

# Caminho base para os Mocks
PATCH_BASE = "app.api.services.drift_detector"

# ==========================================
# 1. TESTES DO GATILHO DO GITHUB ACTIONS
# ==========================================

@patch.dict(os.environ, {"GITHUB_TOKEN": "fake_token", "GITHUB_OWNER": "owner", "GITHUB_REPO": "repo"})
@patch(f"{PATCH_BASE}.requests.post")
def test_disparar_retreino_github_sucesso(mock_post):
    # Finge que o GitHub respondeu com "204 No Content" (Sucesso no disparo do webhook)
    mock_response = MagicMock()
    mock_response.status_code = 204
    mock_post.return_value = mock_response

    # Ação
    disparar_retreino_github("RACE")

    # Verificação: O requests.post foi chamado com os dados certos?
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert "https://api.github.com/repos/owner/repo/actions" in args[0]
    assert kwargs["json"]["inputs"]["symbol"] == "RACE"
    assert kwargs["headers"]["Authorization"] == "token fake_token"

@patch.dict(os.environ, {}, clear=True) # Limpa as variáveis de ambiente
@patch(f"{PATCH_BASE}.requests.post")
def test_disparar_retreino_github_sem_token(mock_post):
    # Se o token faltar, a função deve abortar e NÃO chamar o requests.post
    # Ajustamos um settings falso temporariamente caso ele tente ler de lá
    with patch(f"{PATCH_BASE}.settings") as mock_settings:
        mock_settings.GITHUB_TOKEN = None
        
        disparar_retreino_github("RACE")
        mock_post.assert_not_called()


# ==========================================
# 2. TESTES DA LEITURA DE LOGS DO S3
# ==========================================

@patch(f"{PATCH_BASE}.read_json_from_s3")
@patch(f"{PATCH_BASE}.get_s3_client")
def test_load_production_logs_sucesso(mock_get_s3_client, mock_read_json):
    # Simula o boto3 encontrando 2 arquivos JSON na pasta
    mock_s3 = MagicMock()
    mock_s3.list_objects_v2.return_value = {
        'Contents': [{'Key': 'predictions/RACE/log1.json'}, {'Key': 'predictions/RACE/log2.json'}]
    }
    mock_get_s3_client.return_value = mock_s3
    
    # Simula o conteúdo de cada JSON extraído do S3
    mock_read_json.return_value = {"features_input": {"RSI_14": 50, "Volume": 1000}}
    
    # Ação
    df_logs = load_production_logs("RACE")
    
    # Verificação
    assert not df_logs.empty
    assert len(df_logs) == 2 # Leu os dois logs
    assert "RSI_14" in df_logs.columns


# ==========================================
# 3. TESTES DO MOTOR DE DRIFT (EVIDENTLY AI)
# ==========================================

@patch(f"{PATCH_BASE}.read_csv_from_s3")
def test_check_data_drift_sem_referencia(mock_read_csv):
    # Se o S3 não achar o histórico base, deve abortar com False
    mock_read_csv.return_value = None
    
    resultado = check_data_drift("RACE")
    assert resultado is False

@patch(f"{PATCH_BASE}.load_production_logs")
@patch(f"{PATCH_BASE}.read_csv_from_s3")
def test_check_data_drift_dados_insuficientes(mock_read_csv, mock_load_logs, mock_df_referencia):
    mock_read_csv.return_value = mock_df_referencia
    # Retorna apenas 2 linhas (menos do que o mínimo de 5 que você configurou)
    mock_load_logs.return_value = pd.DataFrame({"RSI_14": [50, 51]})
    
    resultado = check_data_drift("RACE")
    assert resultado is False


# O TESTE SUPREMO: Sem Drift (Caminho Feliz)
@patch("builtins.open", new_callable=mock_open, read_data="<html>Mock HTML</html>")
@patch(f"{PATCH_BASE}.os.remove")
@patch(f"{PATCH_BASE}.os.path.exists") # <--- AQUI ESTÁ O MOCK NOVO
@patch(f"{PATCH_BASE}.write_html_to_s3")
@patch(f"{PATCH_BASE}.Report")
@patch(f"{PATCH_BASE}.load_production_logs")
@patch(f"{PATCH_BASE}.read_csv_from_s3")
@patch(f"{PATCH_BASE}.disparar_retreino_github")
def test_check_data_drift_estavel(mock_github, mock_read_csv, mock_load_logs, mock_report_class, mock_write_s3, mock_exists, mock_remove, mock_file_open, mock_df_referencia, mock_df_producao):
    # Força a função a achar que o HTML existe no disco
    mock_exists.return_value = True
    
    # Setup Mocks (S3)
    mock_read_csv.return_value = mock_df_referencia
    mock_load_logs.return_value = mock_df_producao
    
    # Setup Mock (Evidently Report)
    mock_resultado_eval = MagicMock()
    mock_resultado_eval.dict.return_value = {
        "metrics": [{
            "config": {"type": "DriftedColumnsCount", "drift_share": 0.5},
            "value": {"share": 0.1} # APENAS 10% DE DRIFT (Menor que 0.5 = SEM DRIFT)
        }]
    }
    mock_report_instance = MagicMock()
    mock_report_instance.run.return_value = mock_resultado_eval
    mock_report_class.return_value = mock_report_instance
    
    # Ação
    resultado = check_data_drift("RACE")
    
    # Verificação:
    assert resultado is False # Não tem drift
    mock_write_s3.assert_called_once() # Confirmamos que o HTML do dashboard foi gerado e salvo no S3
    mock_github.assert_not_called() # A esteira NÃO pode ser chamada atoa!
    mock_remove.assert_called_once() # Garante que a limpeza de memória serverless rodou com sucesso


# O TESTE SUPREMO: Com Drift (ALERTA VERMELHO)
@patch("builtins.open", new_callable=mock_open, read_data="<html>Mock HTML</html>")
@patch(f"{PATCH_BASE}.os.remove")
@patch(f"{PATCH_BASE}.os.path.exists") # <--- AQUI ESTÁ O MOCK NOVO
@patch(f"{PATCH_BASE}.write_html_to_s3")
@patch(f"{PATCH_BASE}.Report")
@patch(f"{PATCH_BASE}.load_production_logs")
@patch(f"{PATCH_BASE}.read_csv_from_s3")
@patch(f"{PATCH_BASE}.disparar_retreino_github")
def test_check_data_drift_detectado(mock_github, mock_read_csv, mock_load_logs, mock_report_class, mock_write_s3, mock_exists, mock_remove, mock_file_open, mock_df_referencia, mock_df_producao):
    # Força a função a achar que o HTML existe no disco
    mock_exists.return_value = True
    
    mock_read_csv.return_value = mock_df_referencia
    mock_load_logs.return_value = mock_df_producao
    
    # Setup Mock (Evidently Report com ALERTA)
    mock_resultado_eval = MagicMock()
    mock_resultado_eval.dict.return_value = {
        "metrics": [{
            "config": {"type": "DriftedColumnsCount", "drift_share": 0.5},
            "value": {"share": 0.8} # 80% DE DRIFT! O MUNDO ESTÁ ACABANDO!
        }]
    }
    mock_report_instance = MagicMock()
    mock_report_instance.run.return_value = mock_resultado_eval
    mock_report_class.return_value = mock_report_instance
    
    # Ação
    resultado = check_data_drift("RACE")
    
    # Verificação:
    assert resultado is True # Tem drift!
    mock_github.assert_called_once_with("RACE") # A função salvadora foi chamada!
    mock_remove.assert_called_once() # Garante que apagou o arquivo temporário da AWS Lambda