import pytest
import subprocess
import sys
import os
from unittest.mock import patch, MagicMock

# ==========================================
# HACK DE DIRETÓRIO PARA O PYTEST
# ==========================================
# Adiciona a pasta exata onde os scripts de ML estão ao "radar" do Python.
# Assim, quando o random_search tentar fazer 'import utils_s3', o Pytest vai achar o arquivo!
caminho_ml = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../app/ml/pipeline'))
sys.path.insert(0, caminho_ml)

from app.ml.pipeline.random_search import run_trial, main

# ==========================================
# DEFINIÇÃO DO CAMINHO BASE DOS MOCKS
# ==========================================
# Mude isto de acordo com o local exato do ficheiro no seu projeto
PATCH_BASE = "app.ml.pipeline.random_search"

# ==========================================
# 1. TESTES DA FUNÇÃO DE SUBPROCESSOS (run_trial)
# ==========================================

@patch(f"{PATCH_BASE}.subprocess.check_call")
def test_run_trial_sucesso(mock_check_call):
    # Finge que o script secundário (worker) rodou perfeitamente no terminal
    mock_check_call.return_value = 0
    
    resultado = run_trial(["python", "train_worker.py", "--symbol", "RACE"])
    
    assert resultado is True
    mock_check_call.assert_called_once()

@patch(f"{PATCH_BASE}.subprocess.check_call")
def test_run_trial_falha(mock_check_call):
    # Finge que o script secundário deu um erro fatal (Crash)
    mock_check_call.side_effect = subprocess.CalledProcessError(1, "cmd")
    
    resultado = run_trial(["python", "train_worker.py", "--symbol", "RACE"])
    
    # A função deve capturar o erro e retornar False, em vez de rebentar o orquestrador
    assert resultado is False


# ==========================================
# 2. TESTES DA ORQUESTRAÇÃO PRINCIPAL (main)
# ==========================================

# Criamos uma "Fixture" que gera a resposta do MLflow simulada
@pytest.fixture
def mock_run_mlflow():
    mock_run = MagicMock()
    mock_run.info.run_id = "run_12345"
    # Finge que o RMSE foi de 0.05
    mock_run.data.metrics.get.return_value = 0.05 
    return mock_run


@patch("utils_s3.delete_from_s3") # Importado localmente no script
@patch(f"{PATCH_BASE}.upload_champion_to_s3")
@patch(f"{PATCH_BASE}.os.walk")
@patch(f"{PATCH_BASE}.os.path.exists")
@patch(f"{PATCH_BASE}.artifacts.download_artifacts")
@patch(f"{PATCH_BASE}.MlflowClient")
@patch(f"{PATCH_BASE}.mlflow")
@patch(f"{PATCH_BASE}.run_trial")
def test_main_modelo_com_scaler_sucesso(mock_run_trial, mock_mlflow, mock_client_class, mock_download, mock_exists, mock_walk, mock_upload, mock_delete, mock_run_mlflow):
    """
    Testa o Caminho Feliz para um modelo Deep Learning (LSTM) 
    que exige que o Scaler seja guardado no S3.
    """
    # 1. Configura os Mocks do MLflow
    mock_run_mlflow.data.params.get.return_value = "lstm" # Simulamos que o vencedor foi um LSTM
    
    mock_client = MagicMock()
    mock_client.get_experiment_by_name.return_value = MagicMock(experiment_id="exp_1")
    mock_client.search_runs.return_value = [mock_run_mlflow]
    mock_client_class.return_value = mock_client
    
    # 2. Configura os Mocks do FileSystem
    mock_download.side_effect = ["/tmp/model_dir", "/tmp/scaler_dir"]
    mock_exists.return_value = True # Finge que os ficheiros baixados existem
    
    # Simulamos o os.walk para ele "encontrar" um ficheiro .pth (PyTorch)
    mock_walk.return_value = [("/tmp/model_dir", [], ["modelo.pth"])]
    
    # AÇÃO
    main()
    
    # VERIFICAÇÕES
    # O orquestrador roda 5 vezes por padrão (N_TRIALS = 5)
    assert mock_run_trial.call_count == 5
    
    # Deve ter procurado pelo vencedor no MLflow
    mock_client.search_runs.assert_called_once()
    
    # COMO É LSTM: Deve enviar o modelo E o scaler para o S3 (2 chamadas)
    assert mock_upload.call_count == 2
    # E não pode chamar a função de apagar
    mock_delete.assert_not_called()


@patch("utils_s3.delete_from_s3") 
@patch(f"{PATCH_BASE}.upload_champion_to_s3")
@patch(f"{PATCH_BASE}.os.walk")
@patch(f"{PATCH_BASE}.os.path.exists")
@patch(f"{PATCH_BASE}.artifacts.download_artifacts")
@patch(f"{PATCH_BASE}.MlflowClient")
@patch(f"{PATCH_BASE}.mlflow")
@patch(f"{PATCH_BASE}.run_trial")
def test_main_modelo_sem_scaler(mock_run_trial, mock_mlflow, mock_client_class, mock_download, mock_exists, mock_walk, mock_upload, mock_delete, mock_run_mlflow):
    """
    Testa o Caminho Feliz para um modelo de Árvore (XGBoost) 
    que exige que o Scaler anterior seja APAGADO do S3.
    """
    # Configuramos para o vencedor ser XGBoost
    mock_run_mlflow.data.params.get.return_value = "xgboost" 
    
    mock_client = MagicMock()
    mock_client.get_experiment_by_name.return_value = MagicMock(experiment_id="exp_1")
    mock_client.search_runs.return_value = [mock_run_mlflow]
    mock_client_class.return_value = mock_client
    
    mock_download.side_effect = ["/tmp/model_dir", "/tmp/scaler_dir"]
    mock_exists.return_value = True
    
    # Simula encontrar um ficheiro .xgb
    mock_walk.return_value = [("/tmp/model_dir", [], ["modelo.xgb"])]
    
    # AÇÃO
    main()
    
    # VERIFICAÇÕES
    # COMO É XGBOOST: Só faz o upload do modelo (1 chamada)
    assert mock_upload.call_count == 1
    
    # DEVE chamar a função de apagar o scaler antigo do S3
    mock_delete.assert_called_once()


@patch(f"{PATCH_BASE}.upload_champion_to_s3")
@patch(f"{PATCH_BASE}.MlflowClient")
@patch(f"{PATCH_BASE}.mlflow")
@patch(f"{PATCH_BASE}.run_trial")
def test_main_experimento_sem_resultados(mock_run_trial, mock_mlflow, mock_client_class, mock_upload):
    """
    Garante que o script não rebenta se os treinamentos falharem
    e não houver modelos registados no MLflow.
    """
    # ==========================================
    # CORREÇÃO AQUI 👇: O Mock Ninja!
    # ==========================================
    mock_runs = MagicMock()
    # Finge que a lista é vazia (False) para o if/else
    mock_runs.__bool__.return_value = False 
    # Fornece o método to_list() para não dar erro no print
    mock_runs.to_list.return_value = []

    mock_client = MagicMock()
    mock_client.get_experiment_by_name.return_value = MagicMock(experiment_id="exp_1")
    # Colocamos o nosso mock ninja como retorno
    mock_client.search_runs.return_value = mock_runs
    
    mock_client_class.return_value = mock_client
    
    # Ação
    main()
    
    # Como não encontrou campeão nenhum, NÃO pode tentar enviar nada para o S3
    mock_upload.assert_not_called()