"""
Baseline de Retrieval Textual TF-IDF sobre dataset_short_answer.

Salida: results/dataset_short_answer/retrieval_textual/tfidf_results.json
"""

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

DATASET_PATH = PROJECT_ROOT / "outputs" / "datasets" / "dataset_short_answer.json"
OUTPUT_PATH = (
    RESULTS_DIR
    / "dataset_short_answer"
    / "retrieval_textual"
    / "tfidf_results.json"
)


def retrieve_top1_tfidf(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    texts = [build_query_text(r) for r in records]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(texts)
    sim_matrix: np.ndarray = cosine_similarity(matrix)
    best_idx, best_scores = top1_excluding_self(sim_matrix)

    results = build_results(records, best_idx, best_scores)
    for i, r in enumerate(results):
        r["retrieved_short_answer_es"] = clean_text(
            records[int(best_idx[i])].get("answer_es", "")
        )
        r.pop("retrieved_answer_es", None)
    return results


def main() -> None:
    records = load_dataset(DATASET_PATH)
    print(f"Registros cargados: {len(records)}")
    results = retrieve_top1_tfidf(records)
    save_results(results, OUTPUT_PATH)


if __name__ == "__main__":
    main()
