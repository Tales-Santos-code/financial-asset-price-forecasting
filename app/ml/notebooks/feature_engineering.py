import sys
import os
import polars as pl
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from app.api.services.finance_service import FinanceService # noqa: E402

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, is_training=True):
        self.is_training = is_training

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> pl.DataFrame:
        print(f"Executando Pipeline 100% Polars (Treinamento: {self.is_training})...")
        
        # 1. Ponte de Entrada: yfinance entrega Pandas, transformamos em Polars
        if isinstance(X, pd.DataFrame):
            if X.index.name == 'Date' or 'Date' in X.index.names:
                X = X.reset_index()
            df = pl.from_pandas(X)
        else:
            df = X.clone()
            
        if "Date" in df.columns:
            df = df.with_columns(
                pl.col("Date").dt.replace_time_zone(None)
            ).sort("Date")

        # 2. Bloco de Cálculos (Lags e OBV)
        df = df.with_columns(
            Log_Return=(pl.col("Close") / pl.col("Close").shift(1)).log(),
            OBV_Signal=pl.when(pl.col("Close") > pl.col("Close").shift(1)).then(pl.col("Volume"))
                       .when(pl.col("Close") < pl.col("Close").shift(1)).then(-pl.col("Volume"))
                       .otherwise(0)
        ).with_columns(
            OBV=pl.col("OBV_Signal").cum_sum(),
            Target_Log_Return=pl.col("Log_Return").shift(-1),
            Lag_1=pl.col("Log_Return").shift(1),
            Lag_2=pl.col("Log_Return").shift(2),
            Lag_3=pl.col("Log_Return").shift(3),
            Lag_5=pl.col("Log_Return").shift(5),
            
            SMA_20=pl.col("Close").rolling_mean(window_size=20),
            SMA_50=pl.col("Close").rolling_mean(window_size=50),
            Close_Std_20=pl.col("Close").rolling_std(window_size=20),
            
            Rolling_Std_14=pl.col("Log_Return").rolling_std(window_size=14),
            Vol_SMA_10=pl.col("Volume").rolling_mean(window_size=10),
            
            Month=pl.col("Date").dt.month(),
            DayOfWeek=pl.col("Date").dt.weekday() - 1
        ).with_columns(
            Distancia_SMA_20=(pl.col("Close") / pl.col("SMA_20")) - 1,
            Distancia_SMA_50=(pl.col("Close") / pl.col("SMA_50")) - 1,
            Bollinger_Upper=pl.col("SMA_20") + (2 * pl.col("Close_Std_20")),
            Bollinger_Lower=pl.col("SMA_20") - (2 * pl.col("Close_Std_20")),
            Volume_Shock=pl.col("Volume") / pl.col("Vol_SMA_10"),
            Month_Sin=(2 * np.pi * pl.col("Month") / 12).sin(),
            Month_Cos=(2 * np.pi * pl.col("Month") / 12).cos(),
            Day_Sin=(2 * np.pi * pl.col("DayOfWeek") / 5).sin(),
            Day_Cos=(2 * np.pi * pl.col("DayOfWeek") / 5).cos()
        ).with_columns(
            Bollinger_Width=(pl.col("Bollinger_Upper") - pl.col("Bollinger_Lower")) / pl.col("SMA_20")
        )

        # 3. Limpeza Inteligente e Drop de Nulos
        cols_features = [c for c in df.columns if c not in ['Target_Log_Return', 'Date', 'OBV_Signal']]
        df = df.drop_nulls(subset=cols_features)
        
        if self.is_training:
            df = df.drop_nulls(subset=['Target_Log_Return'])
        
        cols_to_drop = [
            'Open', 'High', 'Low', 'Adj Close', 'Month', 'DayOfWeek', 
            'SMA_20', 'SMA_50', 'Bollinger_Upper', 'Bollinger_Lower', 
            'Vol_SMA_10', 'Close_Std_20', 'OBV_Signal'
        ]
        cols_existentes = [c for c in cols_to_drop if c in df.columns]
        df = df.drop(cols_existentes)
        
        # O RETORNO AGORA É PURO E EXCLUSIVO DO POLARS
        return df
    
if __name__ == "__main__":
    get_data = FinanceService()
    df_history = get_data.get_historical_data(full=True)[0]

    pipeline_ferrari = FeatureEngineering(is_training=True)
    pipeline_ferrari.fit(df_history)

    x_transformado = pipeline_ferrari.transform(df_history)
    
    # Salva o CSV usando a função nativa do Polars
    caminho_csv = os.path.join('app', 'ml', 'pipeline', 'dados_transformados.csv')
    x_transformado.write_csv(caminho_csv)
    print(f"CSV de treino salvo em: {caminho_csv}")

    caminho_pipeline = os.path.join('app', 'ml', 'pipeline', 'pipeline_limpeza.pkl')
    joblib.dump(pipeline_ferrari, caminho_pipeline)
    print(f"Pipeline pronto para uso salvo em: {caminho_pipeline}")