"""
Interfaz común de retrieval.

Cualquier backend (TF-IDF, E5, multimodal en el futuro) implementa esta
interfaz. El servicio no sabe ni le importa cuál está debajo: solo indexa
y busca.
"""

from abc import ABC, abstractmethod

from .corpus import Case


class Retriever(ABC):
    @abstractmethod
    def index(self, cases: list[Case]) -> None:
        """Construir el índice a partir de la base de casos. Se llama una vez al arrancar."""

    @abstractmethod
    def search(
        self, query: str, k: int, exclude_encounter_id: str | None = None
    ) -> list[tuple[int, float]]:
        """
        Devolver los k casos más similares a `query`.

        Retorna una lista de (índice_del_caso, score) ordenada de mayor a menor
        similitud. El score se normaliza a [0, 1].
        """
