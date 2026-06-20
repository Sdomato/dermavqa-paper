"""
Shared utilities for retrieval baselines on dataset_longest_answer.
"""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path(__file__).parent).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "outputs" / "datasets").exists():
            return candidate
    return start


PROJECT_ROOT = find_project_root()
DATASET_PATH = PROJECT_ROOT / "outputs" / "datasets" / "dataset_longest_answer.json"

# Canonical image location: images live under data/iiyi/images_final/ in the
# split subdirs images_{train,valid,test}/. We also fall back to the legacy flat
# data/images/ layout so older local setups keep working.
IMAGES_DIR = PROJECT_ROOT / "data" / "iiyi" / "images_final"
_LEGACY_IMAGES_DIR = PROJECT_ROOT / "data" / "images"
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
_IMAGE_INDEX: dict[str, Path] | None = None


def _build_image_index() -> dict[str, Path]:
    """Index every image file by filename, searching the canonical dir
    recursively (images live in images_{train,valid,test}/) and the legacy
    flat dir. First match wins."""
    index: dict[str, Path] = {}
    for base in (IMAGES_DIR, _LEGACY_IMAGES_DIR):
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTS:
                index.setdefault(path.name, path)
    return index


def resolve_image_path(img_id: str) -> Path | None:
    """Resolve an image filename (e.g. ``IMG_ENC00908_00001.jpg``) to a real
    path on disk, regardless of which split subdir it sits in. Returns None if
    the image is not found. The index is built once and cached."""
    global _IMAGE_INDEX
    if _IMAGE_INDEX is None:
        _IMAGE_INDEX = _build_image_index()
    if img_id in _IMAGE_INDEX:
        return _IMAGE_INDEX[img_id]
    if "." not in img_id:  # caller passed an id without extension
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            hit = _IMAGE_INDEX.get(img_id + ext)
            if hit is not None:
                return hit
    return None


def clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def build_query_text(record: dict[str, Any]) -> str:
    title = clean_text(record.get("query_title_es", ""))
    content = clean_text(record.get("query_content_es", ""))
    return f"{title} {content}".strip()


def load_dataset(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def top1_excluding_self(
    sim_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (best_indices, best_scores) masking diagonal."""
    mat = sim_matrix.copy().astype(float)
    np.fill_diagonal(mat, -np.inf)
    best_idx = np.argmax(mat, axis=1)
    best_scores = mat[np.arange(len(mat)), best_idx]
    return best_idx, best_scores


def build_results(
    records: list[dict[str, Any]],
    best_idx: np.ndarray,
    best_scores: np.ndarray,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for i, record in enumerate(records):
        retrieved = records[int(best_idx[i])]
        results.append(
            {
                "encounter_id": record["encounter_id"],
                "retrieved_encounter_id": retrieved["encounter_id"],
                "similarity_score": round(float(best_scores[i]), 6),
                "retrieved_answer_es": clean_text(retrieved.get("answer_es", "")),
            }
        )
    return results


def save_results(results: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    scores = [r["similarity_score"] for r in results]
    arr = np.array(scores)
    print(f"Guardados {len(results)} resultados en {output_path}")
    print(f"Score coseno — media: {arr.mean():.4f}  min: {arr.min():.4f}  max: {arr.max():.4f}")
