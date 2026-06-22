"""
Baseline de Retrieval Textual con Sentence-BERT sobre dataset_short_answer.

Modelo: paraphrase-multilingual-MiniLM-L12-v2
Salida: results/dataset_short_answer/retrieval_textual/sbert_results.json
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrieval_utils import (
    PROJECT_ROOT,
    RESULTS_DIR,
    build_query_text,
    build_results,
    clean_text,
    load_dataset,
    save_results,
    top1_excluding_self,
)

MODEL_ID = "paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 64
DATASET_PATH = PROJECT_ROOT / "outputs" / "datasets" / "dataset_short_answer.json"
OUTPUT_PATH = (
    RESULTS_DIR
    / "dataset_short_answer"
    / "retrieval_textual"
    / "sbert_results.json"
)


def main() -> None:
    records = load_dataset(DATASET_PATH)
    print(f"Registros cargados: {len(records)}")

    texts = [build_query_text(r) for r in records]

    print(f"Cargando modelo {MODEL_ID}...")
    model = SentenceTransformer(MODEL_ID)

    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    sim_matrix: np.ndarray = embeddings @ embeddings.T
    best_idx, best_scores = top1_excluding_self(sim_matrix)

    results = build_results(records, best_idx, best_scores)
    for i, r in enumerate(results):
        r["retrieved_short_answer_es"] = clean_text(
            records[int(best_idx[i])].get("answer_es", "")
        )
        r.pop("retrieved_answer_es", None)

    save_results(results, OUTPUT_PATH)


if __name__ == "__main__":
    main()
