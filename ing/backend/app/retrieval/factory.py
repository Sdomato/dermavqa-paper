"""Selección del backend de retrieval según configuración."""

from .base import Retriever


def build_retriever(name: str) -> Retriever:
    name = (name or "tfidf").lower()
    if name == "tfidf":
        from .tfidf import TfidfRetriever

        return TfidfRetriever()
    if name == "e5":
        from .e5 import E5Retriever

        return E5Retriever()
    raise ValueError(f"Backend de retrieval desconocido: {name!r} (usar 'tfidf' o 'e5')")
