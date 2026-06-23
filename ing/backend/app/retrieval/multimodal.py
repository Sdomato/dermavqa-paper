"""
Backend de retrieval multimodal (late fusion E5 + BiomedCLIP).

Consume el cache de embeddings precalculado (`outputs/embeddings/case_embeddings.npz`,
generado offline con `scripts/build_case_embeddings.py`). El servicio NO recalcula
los embeddings de los casos: solo carga el cache y, por cada query, embebe el texto
(E5) y la imagen subida (BiomedCLIP) y fusiona.

Fusión, igual que el baseline del paper:
    score = alpha * minmax(text) + (1 - alpha) * minmax(visual)
con alpha=0.6. Si la query no trae imagen, se usa solo el score de texto.

La matemática de retrieval (`fuse_scores`, `_search_with_vectors`) es numpy puro y
testeable sin modelos. El encoding pesado (torch/transformers/open_clip) se importa
de forma perezosa, así que el resto del servicio no depende de esas libs.
"""

import logging
from pathlib import Path

import numpy as np

from .base import Retriever
from .corpus import Case

logger = logging.getLogger("dermaassist")


def _minmax(v: np.ndarray) -> np.ndarray:
    """Normaliza un vector de scores a [0, 1], ignorando -inf/nan."""
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return np.zeros_like(v, dtype=float)
    lo, hi = float(finite.min()), float(finite.max())
    if hi == lo:
        return np.zeros_like(v, dtype=float)
    out = (v - lo) / (hi - lo)
    out[~np.isfinite(v)] = 0.0
    return out


def _l2norm_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def fuse_scores(
    text_scores: np.ndarray,
    visual_scores: np.ndarray | None,
    alpha: float,
    has_image: np.ndarray,
) -> np.ndarray:
    """
    Late fusion de dos vectores de score (numpy puro, testeable sin modelos).

    `alpha` es el peso del texto. Si `visual_scores` es None (query sin imagen),
    devuelve solo el texto normalizado. Los casos sin imagen no pueden matchear
    por la vía visual (se les pone -inf antes de normalizar).
    """
    text_norm = _minmax(text_scores)
    if visual_scores is None:
        return text_norm
    visual = visual_scores.astype(float).copy()
    visual[~has_image] = -np.inf
    visual_norm = _minmax(visual)
    return alpha * text_norm + (1.0 - alpha) * visual_norm


class MultimodalRetriever(Retriever):
    def __init__(self, embeddings_path: str | Path, alpha: float = 0.6) -> None:
        self.embeddings_path = Path(embeddings_path)
        self.alpha = alpha
        self._text_emb: np.ndarray | None = None
        self._visual_emb: np.ndarray | None = None
        self._has_image: np.ndarray | None = None
        self._encounter_ids: list[str] = []
        # Modelos (lazy).
        self._e5_model = None
        self._e5_tokenizer = None
        self._clip_model = None
        self._clip_preprocess = None
        self._device = None

    # ── índice ───────────────────────────────────────────────────────────────
    def index(self, cases: list[Case]) -> None:
        if not self.embeddings_path.exists():
            raise FileNotFoundError(
                f"No existe el cache de embeddings: {self.embeddings_path}. "
                "Generalo con: python ing/backend/scripts/build_case_embeddings.py"
            )
        data = np.load(self.embeddings_path, allow_pickle=True)
        cache_ids = [str(x) for x in data["encounter_ids"]]
        idx_by_id = {eid: i for i, eid in enumerate(cache_ids)}
        cache_text = _l2norm_rows(data["text_emb"].astype(np.float32))
        cache_visual = _l2norm_rows(data["visual_emb"].astype(np.float32))
        cache_has_image = data["has_image"].astype(bool)

        n = len(cases)
        self._text_emb = np.zeros((n, cache_text.shape[1]), dtype=np.float32)
        self._visual_emb = np.zeros((n, cache_visual.shape[1]), dtype=np.float32)
        self._has_image = np.zeros(n, dtype=bool)
        missing = 0
        for i, case in enumerate(cases):
            j = idx_by_id.get(case.encounter_id)
            if j is None:
                missing += 1
                continue
            self._text_emb[i] = cache_text[j]
            self._visual_emb[i] = cache_visual[j]
            self._has_image[i] = cache_has_image[j]
        self._encounter_ids = [c.encounter_id for c in cases]
        if missing:
            logger.warning("%d casos sin embedding en el cache (rankearán al fondo)", missing)

    # ── búsqueda ─────────────────────────────────────────────────────────────
    def _search_with_vectors(
        self,
        q_text: np.ndarray,
        q_visual: np.ndarray | None,
        k: int,
        exclude_encounter_id: str | None = None,
    ) -> list[tuple[int, float]]:
        """Núcleo de retrieval: numpy puro, sin modelos. Testeable con el cache real."""
        if self._text_emb is None:
            raise RuntimeError("El índice no fue construido (llamar index() primero)")
        text_scores = self._text_emb @ q_text
        visual_scores = (self._visual_emb @ q_visual) if q_visual is not None else None
        fused = fuse_scores(text_scores, visual_scores, self.alpha, self._has_image)

        order = np.argsort(fused)[::-1]
        hits: list[tuple[int, float]] = []
        for idx in order:
            if exclude_encounter_id and self._encounter_ids[idx] == exclude_encounter_id:
                continue
            hits.append((int(idx), float(fused[idx])))
            if len(hits) >= k:
                break
        return hits

    def search(
        self,
        query: str,
        k: int,
        exclude_encounter_id: str | None = None,
        query_image_paths: list | None = None,
    ) -> list[tuple[int, float]]:
        q_text = self._encode_text(query)
        q_visual = self._encode_images(query_image_paths) if query_image_paths else None
        return self._search_with_vectors(q_text, q_visual, k, exclude_encounter_id)

    # ── encoders (lazy, deps pesadas) ────────────────────────────────────────
    def _ensure_models(self) -> None:
        if self._e5_model is not None:
            return
        import torch
        from src.multimodal_retrieval import E5_MODEL_ID, _load_clip_model
        from transformers import AutoModel, AutoTokenizer

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._e5_tokenizer = AutoTokenizer.from_pretrained(E5_MODEL_ID)
        self._e5_model = AutoModel.from_pretrained(E5_MODEL_ID).to(self._device).eval()
        self._clip_model, self._clip_preprocess = _load_clip_model(self._device)

    def _encode_text(self, query: str) -> np.ndarray:
        import torch
        from src.multimodal_retrieval import _mean_pool

        self._ensure_models()
        enc = self._e5_tokenizer(
            ["query: " + query], padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            out = self._e5_model(**enc)
        emb = _mean_pool(out.last_hidden_state, enc["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb[0].cpu().numpy().astype(np.float32)

    def _encode_images(self, paths: list) -> np.ndarray | None:
        import torch
        from PIL import Image

        self._ensure_models()
        tensors = []
        for p in paths:
            try:
                tensors.append(self._clip_preprocess(Image.open(p).convert("RGB")))
            except Exception:
                continue
        if not tensors:
            return None
        batch = torch.stack(tensors).to(self._device)
        with torch.no_grad():
            feats = self._clip_model.encode_image(batch)
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        mean = feats.mean(0)
        mean = torch.nn.functional.normalize(mean, p=2, dim=0)
        return mean.cpu().numpy().astype(np.float32)
