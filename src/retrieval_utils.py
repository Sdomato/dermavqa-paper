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
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
_IMAGE_SUBDIRS = ["images_train", "images_valid", "images_test"]


def find_image(image_id: str) -> Path | None:
    """Busca un archivo de imagen en IMAGES_DIR y sus subcarpetas conocidas."""
    direct = IMAGES_DIR / image_id
    if direct.exists():
        return direct
    for subdir in _IMAGE_SUBDIRS:
        candidate = IMAGES_DIR / subdir / image_id
        if candidate.exists():
            return candidate
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
        json.dump(results, f, ensure_ascii=False, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
    scores = [r["similarity_score"] for r in results]
    arr = np.array(scores)
    print(f"Guardados {len(results)} resultados en {output_path}")
    print(f"Score coseno — media: {arr.mean():.4f}  min: {arr.min():.4f}  max: {arr.max():.4f}")
