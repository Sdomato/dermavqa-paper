"""
Baseline de Retrieval Textual TF-IDF sobre dataset_longest_answer.

Para cada caso, recupera el Top-1 más similar (coseno) excluyendo el caso
propio y guarda: encounter_id original, encounter_id recuperado, score de
similitud y respuesta recuperada.

Salida: outputs/results/dataset_longest_answer/retrieval_textual/tfidf_results.json
"""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path(__file__).parent).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "outputs" / "datasets").exists():
            return candidate
    return start


PROJECT_ROOT = find_project_root()
DATASET_PATH = PROJECT_ROOT / "outputs" / "datasets" / "dataset_longest_answer.json"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "results"
    / "dataset_longest_answer"
    / "retrieval_textual"
    / "tfidf_results.json"
)


def clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def build_query_text(record: dict[str, Any]) -> str:
    title = clean_text(record.get("query_title_es", ""))
    content = clean_text(record.get("query_content_es", ""))
    return f"{title} {content}".strip()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def retrieve_top1_tfidf(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    texts: list[str] = [build_query_text(r) for r in records]

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

    # Mask self-similarity so argmax never picks the query itself.
    np.fill_diagonal(sim_matrix, -1.0)

    results: list[dict[str, Any]] = []
    for i, record in enumerate(records):
        best_idx = int(np.argmax(sim_matrix[i]))
        best_score = float(sim_matrix[i, best_idx])
        retrieved = records[best_idx]
        results.append(
            {
                "encounter_id": record["encounter_id"],
                "retrieved_encounter_id": retrieved["encounter_id"],
                "similarity_score": round(best_score, 6),
                "retrieved_answer_es": clean_text(retrieved.get("answer_es", "")),
            }
        )
    return results


def main() -> None:
    print(f"Dataset: {DATASET_PATH}")
    records = load_dataset(DATASET_PATH)
    print(f"Registros cargados: {len(records)}")

    results = retrieve_top1_tfidf(records)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Resultados guardados ({len(results)} casos): {OUTPUT_PATH}")

    scores = [r["similarity_score"] for r in results]
    print(f"Score coseno — media: {np.mean(scores):.4f}  min: {np.min(scores):.4f}  max: {np.max(scores):.4f}")


if __name__ == "__main__":
    main()
