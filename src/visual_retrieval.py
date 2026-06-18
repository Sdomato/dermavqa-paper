"""
Baseline de Retrieval Visual con BiomedCLIP sobre dataset_longest_answer.

Modelo: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 (via open_clip)
Las imágenes se buscan en data/images/ usando los IDs de imagen del campo `image_ids`.
Cuando un caso tiene varias imágenes se promedia su embedding.
Casos sin imagen válida obtienen un embedding cero y no son candidatos de retrieval
(sus scores quedan en -inf al buscar el top-1).

Salida: outputs/results/dataset_longest_answer/retrieval_visual/visual_results.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.retrieval_utils import (
    PROJECT_ROOT,
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

MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
FALLBACK_MODEL = "ViT-B-32"
FALLBACK_PRETRAINED = "openai"
BATCH_SIZE = 32
OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "results"
    / "dataset_longest_answer"
    / "retrieval_visual"
    / "visual_results.json"
)
SIM_MATRIX_CACHE = (
    PROJECT_ROOT
    / "outputs"
    / "results"
    / "dataset_longest_answer"
    / "retrieval_visual"
    / "visual_sim_matrix.npy"
)
HAS_IMAGE_CACHE = (
    PROJECT_ROOT
    / "outputs"
    / "results"
    / "dataset_longest_answer"
    / "retrieval_visual"
    / "visual_has_image.npy"
)


def load_model(
    device: torch.device,
) -> tuple[Any, Any, Any]:
    """Load BiomedCLIP; fall back to ViT-B/32 if hub model unavailable."""
    if not OPEN_CLIP_AVAILABLE:
        raise ImportError("open_clip_torch is required: pip install open_clip_torch")

    try:
        print(f"Cargando {MODEL_ID}...")
        model, _, preprocess = open_clip.create_model_and_transforms(MODEL_ID)
    except Exception as exc:
        print(f"BiomedCLIP no disponible ({exc}). Usando fallback {FALLBACK_MODEL}/{FALLBACK_PRETRAINED}.")
        model, _, preprocess = open_clip.create_model_and_transforms(
            FALLBACK_MODEL, pretrained=FALLBACK_PRETRAINED
        )

    model = model.to(device)
    model.eval()
    return model, preprocess


def resolve_image_paths(image_ids: list[str]) -> list[Path]:
    """Return existing paths for a list of image filenames."""
    paths: list[Path] = []
    for img_id in image_ids:
        candidate = find_image(img_id)
        if candidate is not None:
            paths.append(candidate)
    return paths


def embed_record_images(
    image_paths: list[Path],
    model: Any,
    preprocess: Any,
    device: torch.device,
) -> np.ndarray | None:
    """Return mean-pooled, L2-normalised embedding for a list of image paths, or None."""
    tensors: list[torch.Tensor] = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            tensors.append(preprocess(img))
        except Exception:
            continue

    if not tensors:
        return None

    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        features = model.encode_image(batch)
        features = torch.nn.functional.normalize(features, p=2, dim=1)

    return features.mean(dim=0).cpu().numpy()


def build_image_embeddings(
    records: list[dict[str, Any]],
    model: Any,
    preprocess: Any,
    device: torch.device,
    embed_dim: int,
) -> tuple[np.ndarray, list[bool]]:
    """
    Returns:
        embeddings: (N, D) float32 array — zero vector when no image found.
        has_image:  bool list of length N.
    """
    embeddings = np.zeros((len(records), embed_dim), dtype=np.float32)
    has_image: list[bool] = []

    for i, record in enumerate(tqdm(records, desc="Embedding imágenes")):
        image_ids: list[str] = record.get("image_ids") or []
        paths = resolve_image_paths(image_ids)
        emb = embed_record_images(paths, model, preprocess, device)
        if emb is not None:
            embeddings[i] = emb
            has_image.append(True)
        else:
            has_image.append(False)

    return embeddings, has_image


def get_embed_dim(model: Any, device: torch.device) -> int:
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model.encode_image(dummy)
    return out.shape[-1]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    from src.retrieval_utils import IMAGES_DIR
    if not IMAGES_DIR.exists():
        print(f"ADVERTENCIA: directorio de imágenes no encontrado en {IMAGES_DIR}.")
        print("El script continuará pero todos los embeddings serán cero.")

    records = load_dataset()
    print(f"Registros cargados: {len(records)}")

    model, preprocess = load_model(device)
    embed_dim = get_embed_dim(model, device)
    print(f"Dimensión de embedding: {embed_dim}")

    embeddings, has_image = build_image_embeddings(records, model, preprocess, device, embed_dim)
    n_with_image = sum(has_image)
    print(f"Casos con al menos una imagen válida: {n_with_image}/{len(records)}")

    sim_matrix: np.ndarray = embeddings @ embeddings.T

    no_image_mask = ~np.array(has_image)
    sim_matrix[:, no_image_mask] = -np.inf

    # Cachear para que visual_retrieval_short.py no tenga que recomputar.
    SIM_MATRIX_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(SIM_MATRIX_CACHE, sim_matrix)
    np.save(HAS_IMAGE_CACHE, np.array(has_image))
    print(f"Caché guardada en {SIM_MATRIX_CACHE.parent}")

    best_idx, best_scores = top1_excluding_self(sim_matrix)

    results = build_results(records, best_idx, best_scores)
    for i, r in enumerate(results):
        r["query_has_image"] = has_image[i]

    save_results(results, OUTPUT_PATH)


if __name__ == "__main__":
    main()
