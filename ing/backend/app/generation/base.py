"""Interfaz común de generación de borradores."""

from abc import ABC, abstractmethod
from typing import Any


class Generator(ABC):
    @abstractmethod
    def generate(
        self, query: str, evidence: list[dict[str, Any]], image_paths: list | None = None
    ) -> str:
        """
        Devolver un borrador de respuesta para la consulta, anclado en `evidence`
        (los casos recuperados) y opcionalmente en las imágenes de la query.
        """
