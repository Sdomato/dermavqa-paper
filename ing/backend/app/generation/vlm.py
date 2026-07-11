"""
Generador con el VLM (Qwen2.5-VL-3B + LoRA), opt-in.

Reutiliza la carga de modelo y la generación de `src/vlm_infer.py`. Es pesado
(torch/transformers/peft, GPU recomendada) por eso todo se importa de forma
perezosa y el modelo se carga una sola vez, en el primer borrador.

El prompt RAG (consulta + casos de referencia) se arma con `build_rag_prompt`,
y se manda al modelo junto con las imágenes de la consulta en formato chat de Qwen.
"""

from typing import Any

from .base import Generator
from .prompt import build_rag_prompt

# Mismo system prompt que la inferencia de investigación.
_SYSTEM = (
    "Eres un dermatólogo experto. Analiza la imagen clínica del paciente "
    "y responde la consulta de forma clara y profesional."
)


class VLMGenerator(Generator):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        adapter: str | None = None,
        quantize: str | None = "4bit",
        max_new_tokens: int = 256,
    ) -> None:
        self.model_id = model_id
        self.adapter = adapter or None
        self.quantize = quantize
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from src.vlm_infer import load_model_and_processor

        self._model, self._processor = load_model_and_processor(
            self.model_id, self.quantize, self.adapter
        )

    def generate(
        self, query: str, evidence: list[dict[str, Any]], image_paths: list | None = None
    ) -> str:
        from src.vlm_infer import generate_answer

        self._ensure_model()
        rag_text = build_rag_prompt(query, evidence)
        user_content: list[dict[str, Any]] = [
            {"type": "image", "image": str(p)} for p in (image_paths or [])
        ]
        user_content.append({"type": "text", "text": rag_text})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": _SYSTEM}]},
            {"role": "user", "content": user_content},
        ]
        return generate_answer(self._model, self._processor, messages, self.max_new_tokens)
