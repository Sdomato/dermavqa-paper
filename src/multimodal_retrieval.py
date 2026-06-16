"""
Baseline de Retrieval Multimodal (Late Fusion) sobre dataset_longest_answer.

Combina scores del mejor modelo textual (E5 o SBERT) y del modelo visual (BiomedCLIP)
mediante late fusion:
    final_score = alpha * text_score + (1 - alpha) * visual_score

Por defecto usa E5 como fuente textual y alpha=0.6.
Los scores de cada modalidad se normalizan a [0, 1] antes de fusionarlos.

Requiere que los resultados previos existan:
  outputs/results/dataset_longest_answer/retrieval_textual_e5/e5_results.json
  outputs/results/dataset_longest_answer/retrieval_visual/visual_results.json

Alternativamente puede recalcular los embeddings en memoria si se pasa --recompute.

Salida: outputs/results/dataset_longest_answer/retrieval_multimodal/multimodal_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.retrieval_utils import (
    IMAGES_DIR,
    PROJECT_ROOT,
    build_query_text,
    build_results,
    load_dataset,
    save_results,
    top1_excluding_self,
)

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False

# ── paths ──────────────────────────────────────────────────────────────────────
E5_MODEL_ID = "intfloat/multilingual-e5-base"
BIOMEDCLIP_MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
BIOMEDCLIP_FALLBACK_MODEL = "ViT-B-32"
BIOMEDCLIP_FALLBACK_PRETRAINED = "openai"

TEXT_SIM_CACHE = (
    PROJECT_ROOT / "outputs" / "results" / "dataset_longest_answer"
    / "retrieval_textual_e5" / "e5_sim_matrix.npy"
)
VISUAL_SIM_CACHE = (
    PROJECT_ROOT / "outputs" / "results" / "dataset_longest_answer"
    / "retrieval_visual" / "visual_sim_matrix.npy"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "outputs" / "results" / "dataset_longest_answer"
    / "retrieval_multimodal" / "multimodal_results.json"
)
BATCH_SIZE = 64


# ── helpers ────────────────────────────────────────────────────────────────────

def minmax_norm(matrix: np.ndarray) -> np.ndarray:
    """Normalise a square similarity matrix to [0, 1] row-wise."""
    # Flatten, ignore -inf and nan, compute global min/max.
    flat = matrix.ravel()
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return np.zeros_like(matrix, dtype=float)
    lo, hi = finite.min(), finite.max()
    if hi == lo:
        return np.zeros_like(matrix, dtype=float)
    normed = (matrix - lo) / (hi - lo)
    normed[~np.isfinite(matrix)] = 0.0
    return normed


# ── text embeddings (E5) ───────────────────────────────────────────────────────

def _mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def compute_text_sim_matrix(
    records: list[dict[str, Any]],
    device: torch.device,
) -> np.ndarray:
    texts = ["query: " + build_query_text(r) for r in records]
    tokenizer = AutoTokenizer.from_pretrained(E5_MODEL_ID)
    model = AutoModel.from_pretrained(E5_MODEL_ID).to(device)
    model.eval()

    all_embs: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="E5 text embeddings"):
        enc = tokenizer(
            texts[start : start + BATCH_SIZE],
            padding=True, truncation=True, max_length=512, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(**enc)
        emb = _mean_pool(out.last_hidden_state, enc["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        all_embs.append(emb.cpu().numpy())

    embeddings = np.vstack(all_embs)
    return embeddings @ embeddings.T


# ── visual embeddings (BiomedCLIP) ────────────────────────────────────────────

def _load_clip_model(device: torch.device) -> tuple[Any, Any]:
    if not OPEN_CLIP_AVAILABLE:
        raise ImportError("open_clip_torch is required: pip install open_clip_torch")
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(BIOMEDCLIP_MODEL_ID)
    except Exception as exc:
        print(f"BiomedCLIP no disponible ({exc}). Usando {BIOMEDCLIP_FALLBACK_MODEL}.")
        model, _, preprocess = open_clip.create_model_and_transforms(
            BIOMEDCLIP_FALLBACK_MODEL, pretrained=BIOMEDCLIP_FALLBACK_PRETRAINED
        )
    return model.to(device).eval(), preprocess


def _get_embed_dim(model: Any, device: torch.device) -> int:
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        return model.encode_image(dummy).shape[-1]


def _embed_images_for_record(
    image_ids: list[str],
    model: Any,
    preprocess: Any,
    device: torch.device,
) -> np.ndarray | None:
    from PIL import Image

    tensors: list[torch.Tensor] = []
    for img_id in image_ids:
        p = IMAGES_DIR / img_id
        if not p.exists():
            continue
        try:
            tensors.append(preprocess(Image.open(p).convert("RGB")))
        except Exception:
            continue

    if not tensors:
        return None

    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        feats = model.encode_image(batch)
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
    return feats.mean(0).cpu().numpy()


def compute_visual_sim_matrix(
    records: list[dict[str, Any]],
    device: torch.device,
) -> tuple[np.ndarray, list[bool]]:
    model, preprocess = _load_clip_model(device)
    dim = _get_embed_dim(model, device)
    embeddings = np.zeros((len(records), dim), dtype=np.float32)
    has_image: list[bool] = []

    for i, record in enumerate(tqdm(records, desc="Visual embeddings")):
        emb = _embed_images_for_record(record.get("image_ids") or [], model, preprocess, device)
        if emb is not None:
            embeddings[i] = emb
            has_image.append(True)
        else:
            has_image.append(False)

    sim = embeddings @ embeddings.T
    no_img = ~np.array(has_image)
    sim[:, no_img] = -np.inf
    return sim, has_image


# ── fusion ─────────────────────────────────────────────────────────────────────

def late_fusion(
    text_sim: np.ndarray,
    visual_sim: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Weighted sum of normalised similarity matrices."""
    return alpha * minmax_norm(text_sim) + (1 - alpha) * minmax_norm(visual_sim)


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multimodal Late Fusion retrieval baseline")
    p.add_argument("--alpha", type=float, default=0.6,
                   help="Weight for text score (default: 0.6)")
    p.add_argument("--recompute", action="store_true",
                   help="Recompute similarity matrices even if cache exists")
    return p.parse_args()


def _load_or_compute_text(records: list[dict[str, Any]], device: torch.device, recompute: bool) -> np.ndarray:
    if not recompute and TEXT_SIM_CACHE.exists():
        print(f"Cargando text sim matrix desde cache: {TEXT_SIM_CACHE}")
        return np.load(TEXT_SIM_CACHE)
    print("Calculando text sim matrix (E5)...")
    sim = compute_text_sim_matrix(records, device)
    TEXT_SIM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(TEXT_SIM_CACHE, sim)
    return sim


def _load_or_compute_visual(
    records: list[dict[str, Any]], device: torch.device, recompute: bool
) -> tuple[np.ndarray, list[bool]]:
    if not recompute and VISUAL_SIM_CACHE.exists():
        print(f"Cargando visual sim matrix desde cache: {VISUAL_SIM_CACHE}")
        sim = np.load(VISUAL_SIM_CACHE)
        has_image = [sim[i, i] != -np.inf for i in range(len(records))]
        return sim, has_image
    print("Calculando visual sim matrix (BiomedCLIP)...")
    sim, has_image = compute_visual_sim_matrix(records, device)
    VISUAL_SIM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(VISUAL_SIM_CACHE, sim)
    return sim, has_image


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}  |  alpha (texto): {args.alpha}")

    records = load_dataset()
    print(f"Registros cargados: {len(records)}")

    text_sim = _load_or_compute_text(records, device, args.recompute)
    visual_sim, has_image = _load_or_compute_visual(records, device, args.recompute)

    fused_sim = late_fusion(text_sim, visual_sim, alpha=args.alpha)
    best_idx, best_scores = top1_excluding_self(fused_sim)

    results = build_results(records, best_idx, best_scores)
    for i, r in enumerate(results):
        r["alpha"] = args.alpha
        r["query_has_image"] = has_image[i]

    output = OUTPUT_PATH.parent / f"multimodal_alpha{args.alpha:.2f}_results.json"
    save_results(results, output)


if __name__ == "__main__":
    main()
