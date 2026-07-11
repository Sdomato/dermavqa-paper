"""
Backend de retrieval con Multilingual E5 (calidad del paper).

Mismo modelo y mismo pooling que `src/e5_retrieval.py`. Es más pesado (descarga
el modelo y conviene GPU), por eso torch/transformers se importan de forma
perezosa: el servicio puede correr con TF-IDF sin tener torch instalado.
"""

import numpy as np

from .base import Retriever
from .corpus import Case

MODEL_ID = "intfloat/multilingual-e5-base"
BATCH_SIZE = 64


class E5Retriever(Retriever):
    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._device = None
        self._embeddings: np.ndarray | None = None
        self._encounter_ids: list[str] = []

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self._model = AutoModel.from_pretrained(MODEL_ID).to(self._device).eval()

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        import torch

        self._ensure_model()
        out: list[np.ndarray] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = [prefix + t for t in texts[start : start + BATCH_SIZE]]
            enc = self._tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(self._device)
            with torch.no_grad():
                model_out = self._model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).expand(model_out.last_hidden_state.size()).float()
            pooled = (model_out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.append(pooled.cpu().numpy())
        return np.vstack(out)

    def index(self, cases: list[Case]) -> None:
        # E5 usa prefijo "passage: " para los documentos indexados.
        self._embeddings = self._encode([c.query_text for c in cases], prefix="passage: ")
        self._encounter_ids = [c.encounter_id for c in cases]

    def add(self, new_cases: list[Case], all_cases: list[Case]) -> None:
        """
        Append incremental: embebe solo los casos nuevos y los apila al índice.

        Evita re-embeber los ~1000 casos del corpus en cada aprobación (con E5 eso
        son decenas de segundos). Embeber un caso nuevo es O(1). Si el índice aún
        no existe (cold start), cae al build completo.
        """
        if self._embeddings is None:
            self.index(all_cases)
            return
        nuevos = self._encode([c.query_text for c in new_cases], prefix="passage: ")
        self._embeddings = np.vstack([self._embeddings, nuevos])
        self._encounter_ids = self._encounter_ids + [c.encounter_id for c in new_cases]

    def search(
        self,
        query: str,
        k: int,
        exclude_encounter_id: str | None = None,
        query_image_paths: list | None = None,  # ignorado: backend solo-texto
    ) -> list[tuple[int, float]]:
        if self._embeddings is None:
            raise RuntimeError("El índice no fue construido (llamar index() primero)")

        # E5 usa prefijo "query: " para la consulta.
        q = self._encode([query], prefix="query: ")[0]
        scores = self._embeddings @ q  # coseno (vectores ya normalizados)

        order = scores.argsort()[::-1]
        hits: list[tuple[int, float]] = []
        for idx in order:
            if exclude_encounter_id and self._encounter_ids[idx] == exclude_encounter_id:
                continue
            # Coseno está en [-1, 1]; lo llevamos a [0, 1] para el contrato de la API.
            hits.append((int(idx), float((scores[idx] + 1.0) / 2.0)))
            if len(hits) >= k:
                break
        return hits
