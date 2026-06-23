"""
Configuración del backend, leída de variables de entorno.

Se mantiene simple a propósito (sin dependencias extra): un dataclass con
valores por defecto razonables y override por env var.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Backend de retrieval: "tfidf" (liviano, sin descargas) o "e5" (calidad paper).
    retriever: str = os.getenv("DERMA_RETRIEVER", "tfidf")
    # Cuántos casos similares devolver por consulta.
    top_k: int = int(os.getenv("DERMA_TOP_K", "5"))
    # Qué splits del dataset entran a la base de casos buscable.
    # "all" = los 998 casos; o lista separada por coma, ej: "train".
    index_splits: str = os.getenv("DERMA_INDEX_SPLITS", "all")


settings = Settings()
