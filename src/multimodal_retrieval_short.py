"""
Baseline de Retrieval Multimodal (Late Fusion) sobre dataset_short_answer.

Combina E5 (texto) y BiomedCLIP (visual) mediante late fusion:
    final_score = alpha * norm(text_score) + (1 - alpha) * norm(visual_score)

Los scores de cada modalidad se normalizan a [0, 1] antes de fusionarlos.
Reutiliza los sim-matrix cacheados de los scripts individuales si existen.

Salida:
  results/dataset_short_answer/retrieval_multimodal/
    multimodal_alpha<alpha>_results.json

Uso:
    python -m src.multimodal_retrieval_short
    python -m src.multimodal_retrieval_short --alpha 0.5
    python -m src.multimodal_retrieval_short --recompute
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.retrieval_utils import (
    PROJECT_ROOT,
    RESULTS_DIR,
    build_query_text,
    build_results,
    clean_text,
    find_image,
    load_dataset,
    save_results,
    top1_excluding_self,
)

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False

# ── rutas ──────────────────────────────────────────────────────────────────────

DATASET_PATH = PROJECT_ROOT / "outputs" / "datasets" / "dataset_short_answer.json"
OUTPUT_DIR = (
    RESULTS_DIR / "dataset_short_answer" / "retrieval_multimodal"
)
TEXT_SIM_CACHE = (
    RESULTS_DIR / "dataset_short_answer" / "retrieval_textual" / "e5_sim_matrix.npy"
)
VISUAL_SIM_CACHE = (
    RESULTS_DIR / "dataset_short_answer" / "retrieval_visual" / "visual_sim_matrix.npy"
)

E5_MODEL_ID = "intfloat/multilingual-e5-base"
BIOMEDCLIP_MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
BIOMEDCLIP_FALLBACK_MODEL = "ViT-B-32"
BIOMEDCLIP_FALLBACK_PRETRAINED = "openai"
BATCH_SIZE = 64


# ── normalización ──────────────────────────────────────────────────────────────

def minmax_norm(matrix: np.ndarray) -> np.ndarray:
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


# ── texto (E5) ─────────────────────────────────────────────────────────────────

def _mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def compute_text_sim_matrix(records: list[dict[str, Any]], device: torch.device) -> np.ndarray:
    texts = ["query: " + build_query_text(r) for r in records]
    tokenizer = AutoTokenizer.from_pretrained(E5_MODEL_ID)
    model = AutoModel.from_pretrained(E5_MODEL_ID).to(device)
    model.eval()

    all_embs: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="E5 embeddings"):
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


# ── visual (BiomedCLIP) ────────────────────────────────────────────────────────

def _load_clip(device: torch.device) -> tuple[Any, Any]:
    if not OPEN_CLIP_AVAILABLE:
        raise ImportError("open_clip_torch es requerido: pip install open_clip_torch")
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(BIOMEDCLIP_MODEL_ID)
    except Exception as exc:
        print(f"BiomedCLIP no disponible ({exc}). Usando {BIOMEDCLIP_FALLBACK_MODEL}.")
        model, _, preprocess = open_clip.create_model_and_transforms(
            BIOMEDCLIP_FALLBACK_MODEL, pretrained=BIOMEDCLIP_FALLBACK_PRETRAINED
        )
    return model.to(device).eval(), preprocess


def _embed_dim(model: Any, device: torch.device) -> int:
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        return int(model.encode_image(dummy).shape[-1])


def _embed_record(
    image_ids: list[str], model: Any, preprocess: Any, device: torch.device
) -> np.ndarray | None:
    from PIL import Image

    tensors: list[torch.Tensor] = []
    for img_id in image_ids:
        p = find_image(img_id)
        if p is None:
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
    records: list[dict[str, Any]], device: torch.device
) -> tuple[np.ndarray, list[bool]]:
    model, preprocess = _load_clip(device)
    dim = _embed_dim(model, device)
    embeddings = np.zeros((len(records), dim), dtype=np.float32)
    has_image: list[bool] = []

    for i, record in enumerate(tqdm(records, desc="Visual embeddings")):
        emb = _embed_record(record.get("image_ids") or [], model, preprocess, device)
        if emb is not None:
            embeddings[i] = emb
            has_image.append(True)
        else:
            has_image.append(False)

    sim = embeddings @ embeddings.T
    no_img = ~np.array(has_image)
    sim[:, no_img] = -np.inf
    return sim, has_image


# ── carga con cache ────────────────────────────────────────────────────────────

def _get_text_sim(
    records: list[dict[str, Any]], device: torch.device, recompute: bool
) -> np.ndarray:
    if not recompute and TEXT_SIM_CACHE.exists():
        print(f"Cargando text sim matrix desde cache: {TEXT_SIM_CACHE}")
        return np.load(TEXT_SIM_CACHE)
    print("Calculando text sim matrix (E5)...")
    sim = compute_text_sim_matrix(records, device)
    TEXT_SIM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(TEXT_SIM_CACHE, sim)
    return sim


def _get_visual_sim(
    records: list[dict[str, Any]], device: torch.device, recompute: bool
) -> tuple[np.ndarray, list[bool]]:
    if not recompute and VISUAL_SIM_CACHE.exists():
        print(f"Cargando visual sim matrix desde cache: {VISUAL_SIM_CACHE}")
        sim = np.load(VISUAL_SIM_CACHE)
        has_image = [np.any(np.isfinite(sim[i])) for i in range(len(records))]
        return sim, has_image
    print("Calculando visual sim matrix (BiomedCLIP)...")
    sim, has_image = compute_visual_sim_matrix(records, device)
    VISUAL_SIM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(VISUAL_SIM_CACHE, sim)
    return sim, has_image


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multimodal Late Fusion — dataset_short_answer")
    p.add_argument("--alpha", type=float, default=0.6,
                   help="Peso del score textual (default: 0.6)")
    p.add_argument("--recompute", action="store_true",
                   help="Fuerza recálculo ignorando caché")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}  |  alpha (texto): {args.alpha}")

    records = load_dataset(DATASET_PATH)
    print(f"Registros cargados: {len(records)}")

    text_sim = _get_text_sim(records, device, args.recompute)
    visual_sim, has_image = _get_visual_sim(records, device, args.recompute)

    fused = args.alpha * minmax_norm(text_sim) + (1 - args.alpha) * minmax_norm(visual_sim)
    best_idx, best_scores = top1_excluding_self(fused)

    results = build_results(records, best_idx, best_scores)
    for i, r in enumerate(results):
        r["retrieved_short_answer_es"] = clean_text(
            records[int(best_idx[i])].get("answer_es", "")
        )
        r.pop("retrieved_answer_es", None)
        r["alpha"] = args.alpha
        r["query_has_image"] = has_image[i]

    output_path = OUTPUT_DIR / f"multimodal_alpha{args.alpha:.2f}_results.json"
    save_results(results, output_path)


if __name__ == "__main__":
    main()
