# ruff: noqa: E402
import argparse
import os
import numpy as np
import sys
import codecs

# Garante que a RAIZ do projeto esteja no sys.path para encontrar o pacote 'app'
_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_PIPELINE_DIR, "..", "..", ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

# [VACINA DO EMOJI] Força o Windows a ignorar caracteres que ele não entende (como o 🏃)
# sem quebrar o código. Ele substitui por um '?' internamente.
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('cp1252')(sys.stdout.buffer, 'replace')
    sys.stderr = codecs.getwriter('cp1252')(sys.stderr.buffer, 'replace')
# Garante que o MLflow e outras libs vejam as credenciais da AWS
from app.api.core.config import settings
if settings.AWS_ACCESS_KEY_ID:
    os.environ["AWS_ACCESS_KEY_ID"] = settings.AWS_ACCESS_KEY_ID
if settings.AWS_SECRET_ACCESS_KEY:
    os.environ["AWS_SECRET_ACCESS_KEY"] = settings.AWS_SECRET_ACCESS_KEY
os.environ["AWS_DEFAULT_REGION"] = settings.AWS_REGION

import joblib
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.pytorch
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Importa a nossa função que conversa com o S3
from app.ml.utils.utils_s3 import load_clean_data_from_s3

def create_sequences(data, seq_length, target_index):
    """Cria janelas de tempo garantindo que a coluna Target seja excluída das features."""
    X, y = [], []
    num_features = data.shape[1]
    
    # Cria uma lista de índices com todas as colunas, EXCETO o Target
    feature_indices = [i for i in range(num_features) if i != target_index]

    for i in range(len(data) - seq_length - 1):
        # O modelo APRENDE apenas olhando as colunas que não são o gabarito
        X.append(data[i:(i + seq_length), feature_indices])
        y.append(data[i + seq_length, target_index])
    return np.array(X), np.array(y)

# --- Arquitetura PyTorch (Deep Learning) ---
class SimpleLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]) # Pega apenas o último instante de tempo

class SimpleGRU(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleGRU, self).__init__()
        self.gru = torch.nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

def main():
    # 1. O Ouvinte (Recebe as ordens do random_search.py)
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--sequence_length", type=int, default=24)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--n_estimators", type=int, default=100) # Árvores
    parser.add_argument("--max_depth", type=int, default=5)      # Árvores
    parser.add_argument("--hidden_units", type=int, default=64)  # Redes Neurais
    parser.add_argument("--epochs", type=int, default=20)        # Redes Neurais
    parser.add_argument("--verbose", action="store_true", help="Ativa logs detalhados")
    args = parser.parse_args()

    # 2. Configura o MLflow Local
    from app.api.core.config import settings
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(f"predict_{args.symbol}_v3")

    with mlflow.start_run(run_name=f"{args.model_type}_run"):
        # Loga as configurações sorteadas
        mlflow.log_params(vars(args))

        # ==========================================
        # FASE 1: DADOS E SCALER
        # ==========================================
        try:
            df = load_clean_data_from_s3(args.symbol)
            print(f"Dados carregados para {args.symbol}")
        except Exception as e:
            print(f"Erro ao baixar dados do S3: {e}")
            sys.exit(1)

        # Descobre a posição exata da coluna que queremos prever
        target_col_index = df.columns.get_loc('Target_Log_Return')
        num_features = len(df.columns)
        
        print(f"DEBUG: Dataset shape before scaling: {df.values.shape}")
        dataset = np.nan_to_num(df.values) # Segurança contra divisões por zero do Pandas
        
        #Cria, treina e SALVA o Scaler
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(dataset)
        
        scaler_filename = f"scaler_{args.symbol}.pkl"
        joblib.dump(scaler, scaler_filename)
        mlflow.log_artifact(scaler_filename, artifact_path="scaler") 
        os.remove(scaler_filename) # Limpa o disco local

        # ==========================================
        # FASE 2: PREPARAÇÃO DAS MATRIZES
        # ==========================================
        X, y = create_sequences(scaled_data, args.sequence_length, target_index=target_col_index)
        
        if len(X) < 50:
            print("Dados insuficientes após corte de sequência.")
            return

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # ==========================================
        # FASE 3: TREINAMENTO HÍBRIDO (5 Algoritmos)
        # ==========================================
        
        if args.model_type in ["xgboost", "lightgbm", "random_forest"]:
            # Modelos de árvore precisam dos dados achatados (2D)
            X_train_flat = X_train.reshape((X_train.shape[0], -1))
            X_test_flat = X_test.reshape((X_test.shape[0], -1))
            
            # Corrige o valor -1 para o parâmetro max_depth 
            depth_param = None if args.max_depth == -1 else args.max_depth

            if args.model_type == "xgboost":
                model = xgb.XGBRegressor(
                    n_estimators=args.n_estimators,
                    max_depth=args.max_depth,
                    learning_rate=args.learning_rate,
                    random_state=42
                )
                model.fit(X_train_flat, y_train)
                preds_escaladas = model.predict(X_test_flat)
                mlflow.xgboost.log_model(model, "model")

            elif args.model_type == "lightgbm":
                model = lgb.LGBMRegressor(
                    n_estimators=args.n_estimators,
                    max_depth=args.max_depth,
                    learning_rate=args.learning_rate,
                    random_state=42,
                    verbose=-1 
                )
                model.fit(X_train_flat, y_train)
                preds_escaladas = model.predict(X_test_flat)
                mlflow.lightgbm.log_model(model, "model")

            elif args.model_type == "random_forest":
                model = RandomForestRegressor(
                    n_estimators=args.n_estimators,
                    max_depth=depth_param,
                    random_state=42,
                    n_jobs=-1 
                )
                model.fit(X_train_flat, y_train)
                preds_escaladas = model.predict(X_test_flat)
                mlflow.sklearn.log_model(model, "model")

        elif args.model_type in ["lstm", "gru"]:
            # input_size é o número REAL de features (sem o Target)
            n_input_features = X_train.shape[2]  # (samples, seq_len, features)
            # Inicializa a rede correta
            if args.model_type == "lstm":
                model = SimpleLSTM(input_size=n_input_features, hidden_size=args.hidden_units)
            else:
                model = SimpleGRU(input_size=n_input_features, hidden_size=args.hidden_units)
                
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

            # Converte para Tensores do PyTorch
            X_train_t = torch.tensor(X_train, dtype=torch.float32)
            y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
            X_test_t = torch.tensor(X_test, dtype=torch.float32)

            # Loop de Treino
            model.train()
            for epoch in range(args.epochs):
                optimizer.zero_grad()
                out = model(X_train_t)
                loss = criterion(out, y_train_t)
                loss.backward()
                optimizer.step()
                
                # Log de progresso para evitar timeout e dar feedback
                if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == args.epochs:
                    print(f"      [DL] Época {epoch+1}/{args.epochs} - Loss: {loss.item():.6f}")

            # Predição
            model.eval()
            with torch.no_grad():
                preds_escaladas = model(X_test_t).numpy().flatten()
            
            # Loga o modelo no MLflow
            mlflow.pytorch.log_model(model, "model")

        # ==========================================
        # FASE 4: AVALIAÇÃO NO MUNDO REAL (INVERSE TRANSFORM)
        # ==========================================
        dummy_preds = np.zeros((len(preds_escaladas), num_features))
        dummy_preds[:, target_col_index] = preds_escaladas
        preds_reais = scaler.inverse_transform(dummy_preds)[:, target_col_index]

        dummy_y = np.zeros((len(y_test), num_features))
        dummy_y[:, target_col_index] = y_test
        y_test_reais = scaler.inverse_transform(dummy_y)[:, target_col_index]

        # Calcula as métricas reais
        rmse = np.sqrt(mean_squared_error(y_test_reais, preds_reais))
        mae = mean_absolute_error(y_test_reais, preds_reais)

        # Registra no MLflow
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        
        print(f"Treino concluído! {args.model_type.upper()} -> RMSE Real: {rmse:.5f}")

if __name__ == "__main__":
    main()