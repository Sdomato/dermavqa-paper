"""
Generador stub (sin modelo).

Por defecto el servicio usa este generador: arma un borrador determinístico a
partir de los casos recuperados, sin cargar ningún VLM. Sirve para desarrollar,
demostrar y testear todo el flujo (cola async, endpoints, frontend) sin GPU ni
deps pesadas. El generador real es `VLMGenerator` (opt-in con DERMA_GENERATOR=vlm).
"""

from typing import Any

from .base import Generator


class StubGenerator(Generator):
    def generate(
        self, query: str, evidence: list[dict[str, Any]], image_paths: list | None = None
    ) -> str:
        if not evidence:
            return (
                "[BORRADOR DE PRUEBA — sin VLM]\n"
                "No se encontraron casos similares para fundamentar un borrador. "
                "Configurá DERMA_GENERATOR=vlm para usar el modelo real."
            )
        top = evidence[0]
        n = len(evidence)
        imgs = f" Se recibieron {len(image_paths)} imagen(es)." if image_paths else ""
        return (
            "[BORRADOR DE PRUEBA — sin VLM]\n"
            f"Basado en {n} caso(s) similar(es).{imgs}\n\n"
            f"El caso más parecido sugiere:\n{str(top.get('answer', '')).strip()}\n\n"
            "⚠️ Borrador automático preliminar — requiere revisión y aprobación médica. "
            "Para generar texto real con el modelo, configurá DERMA_GENERATOR=vlm."
        )
