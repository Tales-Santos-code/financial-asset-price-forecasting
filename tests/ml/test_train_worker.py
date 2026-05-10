import pytest
import numpy as np
import pandas as pd
import sys
import os
from unittest.mock import patch, MagicMock

# ==========================================
# HACK DE DIRETÓRIO PARA O PYTEST
# ==========================================
# Adiciona a pasta exata onde os scripts de ML estão ao "radar" do Python.
caminho_ml = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../app/ml/training'))
sys.path.insert(0, caminho_ml)

# Import base
from app.ml.training.train_worker import create_sequences, main

# Caminho base atualizado para os Mocks
PATCH_BASE = "app.ml.training.train_worker"

# ==========================================
# FIXTURES E DADOS FALSOS (MOCKS)
# ==========================================
@pytest.fixture
def mock_df_treino():
    """
    Simula 100 dias de dados de ações com 3 colunas.
    Tem que ter mais de 50 linhas de sobra para não ser barrado no 'if len(X) < 50'.
    """
    np.random.seed(42)
    return pd.DataFrame({
        "RSI_14": np.random.rand(100),
        "Volume": np.random.randint(1000, 5000, 100),
        "Target_Log_Return": np.random.randn(100) # Coluna Obrigatória
    })


# ==========================================
# 1. TESTE UNITÁRIO: PREPARAÇÃO DE DADOS
# ==========================================
def test_create_sequences():
    """Garante que a janelamento de séries temporais está correto"""
    # Matriz 10x2 de mentira
    data = np.array([
        [1, 10], [2, 20], [3, 30], [4, 40], [5, 50],
        [6, 60], [7, 70], [8, 80], [9, 90], [10, 100]
    ])
    
    seq_length = 3
    target_index = 1 # Queremos prever a segunda coluna (índice 1)
    
    X, y = create_sequences(data, seq_length, target_index)
    
    # Se temos 10 linhas e seq=3, devemos gerar (10 - 3 - 1) = 6 sequências
    assert len(X) == 6
    assert len(y) == 6
    
    # A primeira sequência (X) deve ter os dias 1, 2 e 3 — mas SEM a coluna target (índice 1)
    # A função exclui target_index das features, então X[0] tem apenas a coluna 0 (valores 1,2,3)
    assert np.array_equal(X[0], [[1], [2], [3]])
    
    # O alvo (y) do primeiro X deve ser o dia 4 da coluna target_index (valor 40)
    assert y[0] == 40


# ==========================================
# 2. TESTES DE RESILIÊNCIA E REGRAS DE NEGÓCIO
# ==========================================
@patch(f"{PATCH_BASE}.load_clean_data_from_s3")
@patch(f"{PATCH_BASE}.mlflow") # <--- A CORREÇÃO ESTÁ AQUI! Bloqueia a rede do MLflow
def test_main_falha_s3_aborta(mock_mlflow, mock_load_s3):
    """Garante que o script morre graciosamente se o S3 estiver fora do ar"""
    mock_load_s3.side_effect = Exception("AWS Timeout")
    
    # Injeta argumentos pela linha de comando
    test_args = ["train_worker.py", "--symbol", "RACE", "--model_type", "xgboost"]
    
    with patch.object(sys, 'argv', test_args):
        # Como o script usa sys.exit(1), precisamos interceptar isso no Pytest
        with pytest.raises(SystemExit) as excinfo:
            main()
        
        assert excinfo.value.code == 1


@patch(f"{PATCH_BASE}.load_clean_data_from_s3")
@patch(f"{PATCH_BASE}.mlflow") # Bloqueia o MLflow para não tentar logar num servidor inexistente
def test_main_dados_insuficientes(mock_mlflow, mock_load_s3):
    """Garante que a trava de segurança (len(X) < 50) impede treinos viciados"""
    # Passamos um dataframe com apenas 20 linhas!
    mock_load_s3.return_value = pd.DataFrame({
        "Feature": np.random.rand(20),
        "Target_Log_Return": np.random.randn(20)
    })
    
    test_args = ["train_worker.py", "--symbol", "RACE", "--model_type", "xgboost", "--sequence_length", "5"]
    
    with patch.object(sys, 'argv', test_args):
        # A função deve rodar e retornar None (return) antes de treinar o modelo
        resultado = main()
        assert resultado is None


# ==========================================
# 3. SMOKE TESTS - A FÁBRICA FUNCIONA? (O TESTE SUPREMO)
# ==========================================
# Parametrizamos o teste para rodar 5 vezes, uma para CADA ALGORITMO!
@pytest.mark.parametrize("model_type", ["xgboost", "lightgbm", "random_forest", "lstm", "gru"])
@patch(f"{PATCH_BASE}.load_clean_data_from_s3")
@patch(f"{PATCH_BASE}.mlflow") 
@patch(f"{PATCH_BASE}.joblib.dump") 
@patch(f"{PATCH_BASE}.os.remove")
def test_main_treino_completo_todos_modelos(mock_remove, mock_joblib, mock_mlflow, mock_load_s3, mock_df_treino, model_type):
    """
    SMOKE TEST: Garante que os DataFrames do Pandas são achatados (Flatten) perfeitamente para as 
    Árvores e convertidos em Tensores 3D corretos para as Redes Neurais sem dar "Shape Error".
    """
    # Setup
    mock_load_s3.return_value = mock_df_treino
    
    # Argumentos do terminal simulados (Usamos 1 época e poucas árvores só pra ver se o código não quebra)
    test_args = [
        "train_worker.py", 
        "--symbol", "RACE", 
        "--model_type", model_type,
        "--sequence_length", "5", 
        "--epochs", "1",
        "--n_estimators", "2",
        "--hidden_units", "8"
    ]
    
    with patch.object(sys, 'argv', test_args):
        # AÇÃO: Executamos o loop de treino inteiro!
        main()
    
    # VERIFICAÇÕES DE INTEGRIDADE:
    # 1. O Scaler foi salvo no disco e deletado depois (Boa prática na Nuvem)
    mock_joblib.assert_called_once()
    mock_remove.assert_called_once()
    
    # 2. O MLflow registou as métricas de avaliação do mundo real?
    # O mlflow.log_metric tem que ser chamado pelo menos duas vezes (RMSE e MAE)
    assert mock_mlflow.log_metric.call_count >= 2