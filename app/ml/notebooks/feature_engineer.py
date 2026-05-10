# Este módulo existe para compatibilidade com o pipeline serializado (pickle).
# O artefato models/pipeline/pipeline.pkl foi gerado quando este arquivo se chamava
# feature_engineer.py. Re-exportamos tudo de feature_engineer1 para manter compatibilidade.
from app.ml.notebooks.feature_engineer1 import FeatureEngineering  # noqa: F401
