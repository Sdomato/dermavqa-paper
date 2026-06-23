"""
Configuración del backend, leída de variables de entorno.

Se mantiene simple a propósito (sin dependencias extra): un dataclass con
valores por defecto razonables, override por env var y validación fail-fast
(si algo está mal configurado, el servicio no arranca).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

APP_VERSION = "0.6.0"
VALID_RETRIEVERS = {"tfidf", "e5", "multimodal"}
VALID_GENERATORS = {"stub", "vlm"}

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
    # Generación de borradores (Fase 2).
    generator: str = os.getenv("DERMA_GENERATOR", "stub").lower()
    vlm_model_id: str = os.getenv("DERMA_VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
    adapter_path: str = os.getenv("DERMA_ADAPTER_PATH", "")
    max_new_tokens: int = int(os.getenv("DERMA_MAX_NEW_TOKENS", "256"))
    # Casos de evidencia que se pasan al generador como contexto RAG.
    rag_k: int = int(os.getenv("DERMA_RAG_K", "3"))
    # Audit log de revisiones médicas (Fase 3).
    audit_path: str = os.getenv(
        "DERMA_AUDIT_PATH", str(_REPO_ROOT / "ing" / "backend" / ".data" / "revisiones.jsonl")
    )

    def __post_init__(self) -> None:
        if self.retriever not in VALID_RETRIEVERS:
            raise ValueError(
                f"DERMA_RETRIEVER inválido: {self.retriever!r}. "
                f"Opciones: {sorted(VALID_RETRIEVERS)}"
            )
        if self.generator not in VALID_GENERATORS:
            raise ValueError(
                f"DERMA_GENERATOR inválido: {self.generator!r}. "
                f"Opciones: {sorted(VALID_GENERATORS)}"
            )
        if self.top_k < 1:
            raise ValueError(f"DERMA_TOP_K debe ser >= 1 (es {self.top_k})")
        if self.max_k < self.top_k:
            raise ValueError(f"DERMA_MAX_K ({self.max_k}) no puede ser menor que DERMA_TOP_K ({self.top_k})")


settings = Settings()
