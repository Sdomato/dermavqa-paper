"""
Retrieval Visual con BiomedCLIP sobre dataset_short_answer.

Modelo: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 (via open_clip)
        Fallback: ViT-B-32/openai si el hub model no está disponible.

Las imágenes se buscan en data/images/ usando el campo `image_ids`.
Cuando un caso tiene varias imágenes se promedian sus embeddings (mean-pool).
Casos sin imagen válida no son candidatos de retrieval (sus scores quedan en -inf).

Salida: outputs/results/dataset_short_answer/retrieval_visual/visual_results.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.retrieval_utils import (
    IMAGES_DIR,
    PROJECT_ROOT,
    clean_text,
    load_dataset,
    save_results,
    top1_excluding_self,
)

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False

# ── configuración ──────────────────────────────────────────────────────────────

DATASET_PATH = PROJECT_ROOT / "outputs" / "datasets" / "dataset_short_answer.json"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "results"
    / "dataset_short_answer"
    / "retrieval_visual"
    / "visual_results.json"
)

BIOMEDCLIP_MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
FALLBACK_MODEL = "ViT-B-32"
FALLBACK_PRETRAINED = "openai"

# ── carga del modelo ───────────────────────────────────────────────────────────


def load_clip_model(device: torch.device) -> tuple[Any, Any]:
    if not OPEN_CLIP_AVAILABLE:
        raise ImportError(
            "open_clip_torch no está instalado. Ejecuta: pip install open_clip_torch"
        )
    try:
        print(f"Cargando {BIOMEDCLIP_MODEL_ID}...")
        model, _, preprocess = open_clip.create_model_and_transforms(BIOMEDCLIP_MODEL_ID)
    except Exception as exc:
        print(f"BiomedCLIP no disponible ({exc}).")
        print(f"Usando fallback: {FALLBACK_MODEL}/{FALLBACK_PRETRAINED}")
        model, _, preprocess = open_clip.create_model_and_transforms(
            FALLBACK_MODEL, pretrained=FALLBACK_PRETRAINED
        )
    return model.to(device).eval(), preprocess


def get_embed_dim(model: Any, device: torch.device) -> int:
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        return int(model.encode_image(dummy).shape[-1])


# ── embedding de imágenes ──────────────────────────────────────────────────────


def embed_record_images(
    image_ids: list[str],
    model: Any,
    preprocess: Any,
    device: torch.device,
) -> np.ndarray | None:
    """
    Devuelve el embedding L2-normalizado y mean-pooled de todas las imágenes
    válidas de un caso. Retorna None si ninguna imagen existe o puede abrirse.
    """
    tensors: list[torch.Tensor] = []
    for img_id in image_ids:
        path = IMAGES_DIR / img_id
        if not path.exists():
            continue
        try:
            tensors.append(preprocess(Image.open(path).convert("RGB")))
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
    Retorna:
        embeddings : (N, D) float32 — vector cero cuando no hay imagen válida.
        has_image  : lista bool de longitud N.
    """
    embeddings = np.zeros((len(records), embed_dim), dtype=np.float32)
    has_image: list[bool] = []

    for i, record in enumerate(tqdm(records, desc="Embedding imágenes")):
        image_ids: list[str] = record.get("image_ids") or []
        emb = embed_record_images(image_ids, model, preprocess, device)
        if emb is not None:
            embeddings[i] = emb
            has_image.append(True)
        else:
            has_image.append(False)

    return embeddings, has_image


# ── construcción de resultados ─────────────────────────────────────────────────


def build_results_short(
    records: list[dict[str, Any]],
    best_idx: np.ndarray,
    best_scores: np.ndarray,
    has_image: list[bool],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for i, record in enumerate(records):
        retrieved = records[int(best_idx[i])]
        results.append(
            {
                "encounter_id": record["encounter_id"],
                "retrieved_encounter_id": retrieved["encounter_id"],
                "similarity_score": round(float(best_scores[i]), 6),
                "retrieved_short_answer_es": clean_text(retrieved.get("answer_es", "")),
                "query_has_image": has_image[i],
            }
        )
    return results


# ── main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    if not IMAGES_DIR.exists():
        print(
            f"ADVERTENCIA: directorio de imágenes no encontrado en {IMAGES_DIR}.\n"
            "Crea data/images/ y coloca allí los archivos IMG_ENC*.jpg.\n"
            "El script continúa pero todos los embeddings serán cero."
        )

    records = load_dataset(DATASET_PATH)
    print(f"Registros cargados: {len(records)}")

    model, preprocess = load_clip_model(device)
    embed_dim = get_embed_dim(model, device)
    print(f"Dimensión de embedding: {embed_dim}")

    embeddings, has_image = build_image_embeddings(records, model, preprocess, device, embed_dim)

    n_with_image = sum(has_image)
    print(f"Casos con al menos una imagen válida: {n_with_image}/{len(records)}")

    sim_matrix: np.ndarray = embeddings @ embeddings.T

    # Los casos sin imagen nunca deben ser recuperados como top-1.
    no_image_mask = ~np.array(has_image)
    sim_matrix[:, no_image_mask] = -np.inf

    best_idx, best_scores = top1_excluding_self(sim_matrix)

    results = build_results_short(records, best_idx, best_scores, has_image)
    save_results(results, OUTPUT_PATH)


if __name__ == "__main__":
    main()
