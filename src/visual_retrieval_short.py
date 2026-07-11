"""
Baseline de Retrieval Visual sobre dataset_short_answer.

Reutiliza la sim matrix cacheada por visual_retrieval.py (las imágenes son
las mismas en ambos datasets). Solo cambia la respuesta que se guarda en el
output: usa answer_es de dataset_short_answer en lugar del longest_answer.

Requiere haber corrido visual_retrieval.py primero.

Salida: results/dataset_short_answer/retrieval_visual/visual_results.json
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.retrieval_utils import (
    PROJECT_ROOT,
    RESULTS_DIR,
    clean_text,
    load_dataset,
    save_results,
    top1_excluding_self,
)

DATASET_PATH = PROJECT_ROOT / "outputs" / "datasets" / "dataset_short_answer.json"
OUTPUT_PATH = (
    RESULTS_DIR
    / "dataset_short_answer"
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


def main() -> None:
    if not SIM_MATRIX_CACHE.exists():
        print("ERROR: caché no encontrada. Corré visual_retrieval.py primero.")
        return

    print(f"Cargando sim matrix desde {SIM_MATRIX_CACHE}...")
    sim_matrix: np.ndarray = np.load(SIM_MATRIX_CACHE)
    has_image: list[bool] = np.load(HAS_IMAGE_CACHE).tolist()

    records = load_dataset(DATASET_PATH)
    print(f"Registros cargados: {len(records)}")

    best_idx, best_scores = top1_excluding_self(sim_matrix)

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

    save_results(results, OUTPUT_PATH)


if __name__ == "__main__":
    main()
