"""
Configuración del backend, leída de variables de entorno.

Se mantiene simple a propósito (sin dependencias extra): un dataclass con
valores por defecto razonables, override por env var y validación fail-fast
(si algo está mal configurado, el servicio no arranca).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

APP_VERSION = "0.3.0"
VALID_RETRIEVERS = {"tfidf", "e5", "multimodal"}

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_EMB = _REPO_ROOT / "outputs" / "embeddings" / "case_embeddings.npz"


@dataclass(frozen=True)
class Settings:
    # Backend de retrieval: "tfidf" (liviano, sin descargas) o "e5" (calidad paper).
    retriever: str = os.getenv("DERMA_RETRIEVER", "tfidf").lower()
    # Cuántos casos similares devolver por consulta (default).
    top_k: int = int(os.getenv("DERMA_TOP_K", "5"))
    # Tope duro de k que un cliente puede pedir.
    max_k: int = int(os.getenv("DERMA_MAX_K", "50"))
    # Qué splits del dataset entran a la base buscable ("all" o ej. "train").
    index_splits: str = os.getenv("DERMA_INDEX_SPLITS", "all")
    # Orígenes permitidos por CORS (coma-separados, "*" para todos en dev).
    cors_origins: list[str] = field(
        default_factory=lambda: [
            o.strip() for o in os.getenv("DERMA_CORS_ORIGINS", "*").split(",") if o.strip()
        ]
    )
    # Multimodal: ruta del cache de embeddings y peso del texto en la fusión.
    embeddings_path: str = os.getenv("DERMA_EMBEDDINGS_PATH", str(_DEFAULT_EMB))
    alpha_text: float = float(os.getenv("DERMA_ALPHA_TEXT", "0.6"))

    def __post_init__(self) -> None:
        if self.retriever not in VALID_RETRIEVERS:
            raise ValueError(
                f"DERMA_RETRIEVER inválido: {self.retriever!r}. "
                f"Opciones: {sorted(VALID_RETRIEVERS)}"
            )
        if self.top_k < 1:
            raise ValueError(f"DERMA_TOP_K debe ser >= 1 (es {self.top_k})")
        if self.max_k < self.top_k:
            raise ValueError(f"DERMA_MAX_K ({self.max_k}) no puede ser menor que DERMA_TOP_K ({self.top_k})")


settings = Settings()
