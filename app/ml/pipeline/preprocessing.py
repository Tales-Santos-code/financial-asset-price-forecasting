# app/ml/pipeline/train_pipeline.py
import os
import sys
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from app.api.services.finance_service import FinanceService # noqa: E402
from app.ml.notebooks.feature_engineer1 import FeatureEngineering # noqa: E402

if __name__ == "__main__":
    get_data = FinanceService()
    
    print("📥 Puxando dados crus...")
    df_history = get_data.get_historical_data(full=True)
    
    min_date = df_history.index.min() if 'Date' not in df_history.columns else df_history['Date'].min()
    max_date = df_history.index.max() if 'Date' not in df_history.columns else df_history['Date'].max()
    df_macro = get_data.get_macro_data(min_date=min_date, max_date=max_date)

    
    pipeline_ferrari = FeatureEngineering(is_training=True)
    
    x_transformado = pipeline_ferrari.fit_transform((df_history, df_macro))
    
    caminho_csv = os.path.join(BASE_DIR, 'app', 'ml', 'pipeline', 'dados_transformados_V2.csv')
    x_transformado.to_csv(caminho_csv, index=False)
    print(f"\n📊 CSV de treino salvo em: {caminho_csv}")

    caminho_pipeline = os.path.join(BASE_DIR, 'app', 'ml', 'pipeline', 'pipeline_limpeza_V2.pkl')
    joblib.dump(pipeline_ferrari, caminho_pipeline)
    print(f"📦 Pipeline salvo em: {caminho_pipeline}")