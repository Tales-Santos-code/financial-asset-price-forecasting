#%%
from logging import config
from flask import config
import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

import sys

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
import xgboost as xgb
import lightgbm as lgb

#%%
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

# 1. Pega a pasta exata onde o seu Jupyter está rodando agora
diretorio_atual = os.getcwd()

# 2. Corta o caminho até a raiz do projeto (antes da pasta 'app')
if "app" in diretorio_atual:
    BASE_DIR = diretorio_atual.split("app")[0]
else:
    BASE_DIR = diretorio_atual # Assume que já está na raiz

#%%
mlflow.set_tracking_uri("http://localhost:5000")  # URL do servidor MLflow local
mlflow.set_experiment("RaceFinancialPredict")  # Nome da experiência (pode ser criado dinamicamente)
mlflow.autolog()  # Ativa o autolog para capturar automaticamente métricas, parâmetros e artefatos

#%%
with mlflow.start_run():

    #Carregar os Dados Transformados
    caminho_csv = os.path.join('app', 'ml', 'pipeline', 'dados_transformados.csv')
    df_tratado = pd.read_csv(caminho_csv)

    y = df_tratado["Target_Log_Return"].values 
    
    # Deletamos as colunas indesejadas mantendo o resto como o DataFrame X
    X = df_tratado.drop(columns=["Target_Log_Return", "Date"]) 

    dias_teste = 90
    train_size = len(X) - dias_teste
    
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    tscv = TimeSeriesSplit(n_splits=5)

    
    modelo = lgb.LGBMRegressor(random_state=42, n_jobs=-1)
    params = {
                "n_estimators": [100, 300, 500],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 8],
                "num_leaves": [20, 31, 50]
            }
    
    mlflow.set_tag("model", type(modelo).__name__)  # Rastreia o nome do modelo como tag no MLflow
    # Usamos RandomizedSearch para economizar tempo computacional 
    search = RandomizedSearchCV(
        estimator=modelo,
        param_distributions=params,
        n_iter=10, 
        scoring='neg_mean_absolute_error', 
        cv=tscv, 
        verbose=1,
        random_state=42,
        n_jobs=-1 
    )

    search.fit(X_train, y_train)

    modelo_campeao = search.best_estimator_
    melhores_params = search.best_params_

    mlflow.log_params(melhores_params)
    mlflow.log_param("best_model", type(modelo_campeao).__name__)
    


    # Extrai o modelo Campeão (A melhor combinação de parâmetros encontrada)
    modelo_campeao = search.best_estimator_
    melhores_params = search.best_params_
    
    print(f"✅ Melhores parâmetros encontrados: {melhores_params}")
    
    # Rastreia os parâmetros vencedores no MLflow
    mlflow.log_params(melhores_params)
    mlflow.log_param("model_type", nome_modelo)

    # 8. Validação Final (Prova de Fogo no arquivo de Teste - Últimos 90 dias)
    preds = modelo_campeao.predict(X_test)




## metrics
# MAE (Mean Absolute Error - Erro Médio Absoluto)
# RMSE (Root Mean Squared Error - Raiz do Erro Quadrático Médio)
# R2 Score (Coeficiente de Determinação)
# MAPE (Mean Absolute Percentage Error - Erro Percentual Absoluto Médio)
# Hit Rate (Taxa de Acerto em Previsões de Direção) (np.sign(y_test) == np.sign(preds)).mean() * 100
# %%
