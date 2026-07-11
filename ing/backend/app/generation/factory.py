"""Selección del generador de borradores según configuración."""

from ..config import settings
from .base import Generator


def build_generator(name: str) -> Generator:
    name = (name or "stub").lower()
    if name == "stub":
        from .stub import StubGenerator

        return StubGenerator()
    if name == "vlm":
        from .vlm import VLMGenerator

        return VLMGenerator(
            model_id=settings.vlm_model_id,
            adapter=settings.adapter_path or None,
            max_new_tokens=settings.max_new_tokens,
        )
    raise ValueError(f"Generador desconocido: {name!r} (usar 'stub' o 'vlm')")
