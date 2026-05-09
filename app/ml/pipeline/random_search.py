import os
import random
import subprocess
import sys
import mlflow
from mlflow.tracking import MlflowClient
from mlflow import artifacts

# Importando as funções do seu utilitário S3 (incluindo a nova função de faxina)
from utils_s3 import upload_champion_to_s3

# Tenta pegar do .env. Se não achar, usa a pasta atual de onde o script está rodando
BASE_DIR = os.getenv("BASE_DIR", os.path.dirname(os.path.abspath(__file__)))

def run_trial(cmd):
    """Executa o script do operário escondendo os logs poluidors do terminal."""
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("Iniciando Fábrica de Experimentos (5 Modelos - ML & DL)...")
    
    # symbols = ["RACE", "NVDA", "AAPL", "VALE3.SA", "ITSA4.SA", "WEGE3.SA", "GSPC"]
    symbols = ["RACE"]
    
    # ==========================================
    # CAIXAS DE FERRAMENTAS CONDICIONAIS
    # Cada modelo tem seus próprios hiperparâmetros
    # ==========================================
    search_spaces = {
        "xgboost": {
            "learning_rate": [0.1, 0.05, 0.01],
            "n_estimators": [100, 300, 500],
            "max_depth": [3, 5, 7, 10, 15], 
            "sequence_length": [24]
        },
        "lightgbm": {
            "learning_rate": [0.1, 0.05, 0.01],
            "n_estimators": [100, 200, 400, 500],
            "max_depth": [5, 10, -1],
            "sequence_length": [24]
        },
        "random_forest": {
            "n_estimators": [100, 300, 500],
            "max_depth": [5, 10, 20, 30],
            "learning_rate": [0.0],
            "sequence_length": [24]
        },
        "lstm": {
            "learning_rate": [0.01, 0.005, 0.001],
            "hidden_units": [32, 64, 128],
            "epochs": [20, 50, 100],
            "sequence_length": [12, 24, 30]
        },
        "gru": {
            "learning_rate": [0.01, 0.005, 0.001],
            "hidden_units": [64, 128, 256],
            "epochs": [30, 60, 90],
            "sequence_length": [12, 24, 30]
        }
    }

    N_TRIALS = 5  # Vai rodar N sorteios no total por Ação
    script_path = os.path.join(os.path.dirname(__file__), "train_worker.py")

    # ==========================================
    # 1. ORQUESTRAÇÃO DOS TREINAMENTOS
    # ==========================================
    for symbol in symbols:
        print(f"\nDisparando {N_TRIALS} experimentos aleatórios para {symbol}...")
        for i in range(N_TRIALS):
            # 1. Sorteia qual será o algoritmo da vez
            model_type = random.choice(list(search_spaces.keys()))
            
            # 2. Sorteia os parâmetros ESPECÍFICOS daquele algoritmo
            p = {k: random.choice(v) for k, v in search_spaces[model_type].items()}
            
            # 3. Cria parâmetros base para o Argparse do worker não quebrar
            base_params = {
                "sequence_length": 24, "learning_rate": 0.01, "n_estimators": 100, 
                "max_depth": 5, "hidden_units": 64, "epochs": 20
            }
            base_params.update(p)

            cmd = [
                sys.executable, script_path,
                "--symbol", symbol,
                "--model_type", model_type,
                "--sequence_length", str(base_params["sequence_length"]),
                "--learning_rate", str(base_params["learning_rate"]),
                "--n_estimators", str(base_params["n_estimators"]),
                "--max_depth", str(base_params["max_depth"]),
                "--hidden_units", str(base_params["hidden_units"]),
                "--epochs", str(base_params["epochs"])
            ]
            
            print(f"   Teste {i+1}/{N_TRIALS} -> Algoritmo: {model_type.upper()} | Config: {p}")
            run_trial(cmd)

    # ==========================================
    # 2. BUSCA DO CAMPEÃO E DEPLOY PARA O S3
    # ==========================================
    print("\nBuscando o Grande Campeão no MLflow...")
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()

    # Criamos uma pasta temporária segura no Windows para os downloads
    temp_download_dir = os.path.join(BASE_DIR, "temp_artifacts")
    os.makedirs(temp_download_dir, exist_ok=True)
    
    # Coleta o nome do bucket do ambiente ou define fixo
    BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "financial-asset-price-forecasting-495599733085-us-east-1-an")

    for symbol in symbols:
        experiment = client.get_experiment_by_name(f"predict_{symbol}")
        print("nome do experimento: ",experiment.name if experiment else f"Experimento para {symbol} não encontrado.") # Debug para verificar se o experimento existe
        if not experiment:
            continue
            
        # Ordena todos os testes pelo Menor RMSE
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.rmse ASC"],
            max_results=1
        )

        
        if runs:
            best_run = runs[0]
            # best_rmse = best_run.data.metrics.get('rmse', 'N/A')
            best_model_type = best_run.data.params.get('model_type', 'N/A')
            run_id = best_run.info.run_id

           
            # Pega o valor bruto
            raw_rmse = best_run.data.metrics.get('rmse', 'N/A')
            
            # Formata bonito apenas se for um número, senão exibe como texto normal
            if isinstance(raw_rmse, (int, float)):
                rmse_formatado = f"{raw_rmse:.5f}"
            else:
                rmse_formatado = str(raw_rmse)

            print(f"🥇 Vencedor Absoluto ({symbol}): {best_model_type.upper()} com RMSE: {rmse_formatado}")
            print("📥 Baixando artefatos do MLflow Local (S3)...")
            
            # Baixa a pasta inteira do modelo e do scaler temporariamente para a máquina local
            local_model_dir = artifacts.download_artifacts(
                run_id=run_id, 
                artifact_path="model", 
                dst_path=temp_download_dir
            )
            
            local_scaler_dir = artifacts.download_artifacts(
                run_id=run_id, 
                artifact_path="scaler", 
                dst_path=temp_download_dir
            )

            # Aponta para os arquivos físicos dentro das pastas baixadas
            arquivo_scaler_local = os.path.join(local_scaler_dir, "scaler.pkl")
            
            # ==========================================
            # LÓGICA DE SOBRESCRITA E FAXINA S3 (COM PASTAS)
            # ==========================================
            
            # 1. Busca inteligente do arquivo físico do modelo na pasta do MLflow
            arquivo_modelo_real = None
            for root, dirs, files in os.walk(local_model_dir):
                for file in files:
                    if file.endswith(".pkl") or file.endswith(".pth") or file.endswith(".xgb"):
                        print(f"🔍 Encontrado arquivo de modelo: {file} na pasta {root}")
                        arquivo_modelo_real = os.path.join(root, file)
                        break
                if arquivo_modelo_real:
                    break

            # 2. Upload do Modelo Principal (Sempre com nome fixo!)
            nome_modelo_s3 = "models/champion/modelo.pkl"
            
            if arquivo_modelo_real and os.path.exists(arquivo_modelo_real):
                print(f"☁️ Subindo o arquivo {arquivo_modelo_real} para o S3 como: {nome_modelo_s3}")
                upload_champion_to_s3(arquivo_modelo_real, nome_modelo_s3)
                print("✅ Upload do modelo campeão concluído com sucesso!")
            else:
                print(f"❌ ERRO GRAVE: Nenhum arquivo de modelo encontrado na pasta {local_model_dir}")
                print(f"📂 O que o MLflow baixou foi: {os.listdir(local_model_dir)}")
            
            # 3. Avalia a necessidade do Scaler e salva na pasta scaler
            nome_scaler_s3 = "models/scaler/scaler.pkl"
            modelos_com_scaler = ['lstm', 'gru']
            
            if best_model_type in modelos_com_scaler and os.path.exists(arquivo_scaler_local):
                print(f"☁️ Sobrescrevendo o Scaler no S3: {nome_scaler_s3}")
                upload_champion_to_s3(arquivo_scaler_local, nome_scaler_s3)
                print("✅ Upload do scaler concluído com sucesso!")
            else:
                print(f"🧹 Modelo é {best_model_type.upper()}. Apagando scaler obsoleto do S3...")
                from utils_s3 import delete_from_s3
                delete_from_s3(BUCKET_NAME, nome_scaler_s3)
        else:
            print(f"⚠️ Nenhum teste encontrado para {symbol}. Verifique se os experimentos rodaram corretamente.")
            print(runs.to_list())

    print("\n🚀 Pipeline de Auto-ML Finalizado!")

if __name__ == "__main__":
    main()