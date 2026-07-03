"""
Pre-computa contextos RAG para entrenamiento RAG-aware.

Para cada fila del dataset (train / valid / test) recupera los top-k casos más
similares del split train usando E5-small multilingual y guarda el resultado
como JSONL con el campo `rag_contexts`.

Regla de leave-one-out en train: cada fila excluye su propio encounter_id del
índice de recuperación — así el modelo no ve su propia respuesta como contexto.

Outputs (por dataset):
  outputs/datasets/<dataset>_rag_e5_small.jsonl   ← train + valid + test aumentados

Uso:
    # Dataset enriched
    python -m src.build_rag_training_dataset --dataset enriched

    # Dataset longest_answer
    python -m src.build_rag_training_dataset --dataset longest

    # Ambos
    python -m src.build_rag_training_dataset --dataset enriched --top-k 3
    python -m src.build_rag_training_dataset --dataset longest  --top-k 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.vlm_by_image_utils import (
    DATASETS_DIR,
    PROJECT_ROOT,
    ByImageDatasetConfig,
    ImageResolver,
    build_inference_items,
    clean_text,
    filter_split,
    load_by_image_dataset,
)
from src.vlm_infer_enriched import CONFIG as ENRICHED_CONFIG
from src.vlm_infer_longest_by_image import CONFIG as LONGEST_CONFIG

E5_MODEL_ID = "intfloat/multilingual-e5-small"

DATASET_CONFIGS: dict[str, ByImageDatasetConfig] = {
    "enriched": ENRICHED_CONFIG,
    "longest": LONGEST_CONFIG,
}

OUT_NAMES: dict[str, str] = {
    "enriched": "dermavqa_iiyi_llm_synthesized_answer_finetune_rag_e5_small.jsonl",
    "longest": "dataset_longest_answer_by_image_rag_e5_small.jsonl",
}


# ── Retriever E5-small ────────────────────────────────────────────────────────

class E5Retriever:
    def __init__(self, questions: list[str]) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(E5_MODEL_ID)
        corpus = [f"passage: {q}" for q in questions]
        print(f"  Codificando {len(corpus)} preguntas del corpus...")
        self.matrix = self.model.encode(
            corpus,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=True,
        )

    def search(self, query: str) -> np.ndarray:
        emb = self.model.encode(
            [f"query: {query}"],
            normalize_embeddings=True,
            batch_size=1,
            show_progress_bar=False,
        )
        return np.matmul(emb, self.matrix.T)[0]


# ── Lógica de recuperación ────────────────────────────────────────────────────

def unique_by_encounter(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        eid = clean_text(item.get("encounter_id", ""))
        if eid and eid not in seen:
            seen.add(eid)
            result.append(item)
    return result


def retrieve_contexts(
    query_item: dict[str, Any],
    retriever: E5Retriever,
    index_items: list[dict[str, Any]],
    top_k: int,
    exclude_encounter_id: str | None = None,
) -> list[dict[str, Any]]:
    scores = retriever.search(query_item["question_es"])
    ranked = scores.argsort()[::-1]

    contexts: list[dict[str, Any]] = []
    for raw_idx in ranked:
        idx = int(raw_idx)
        candidate = index_items[idx]
        if exclude_encounter_id and clean_text(candidate.get("encounter_id", "")) == exclude_encounter_id:
            continue
        answer = clean_text(candidate.get("reference_answer_es", ""))
        if not answer:
            continue
        contexts.append({
            "encounter_id": candidate["encounter_id"],
            "question_es": candidate["question_es"],
            "answer_es": answer,
            "score": round(float(scores[idx]), 6),
        })
        if len(contexts) >= top_k:
            break
    return contexts


# ── Pipeline principal ────────────────────────────────────────────────────────

def load_all_items(
    config: ByImageDatasetConfig,
    dataset_path: Path | None,
) -> dict[str, list[dict[str, Any]]]:
    all_records = load_by_image_dataset(dataset_path, config.default_paths, config.missing_message)
    resolver = ImageResolver()
    splits: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "valid", "test"):
        try:
            records = filter_split(all_records, split, config.split_aliases, config.dataset_name)
        except ValueError:
            continue
        splits[split] = build_inference_items(records, config.answer_columns, resolver)
    return splits


def build_rag_dataset(
    config: ByImageDatasetConfig,
    dataset_path: Path | None,
    top_k: int,
    out_path: Path,
) -> None:
    print(f"\n[{config.dataset_variant}] Cargando splits...")
    splits = load_all_items(config, dataset_path)

    train_items = splits.get("train", [])
    if not train_items:
        raise ValueError(f"No hay items de train en {config.dataset_variant}")

    # Índice único por encounter para retrieval
    index_items = unique_by_encounter(train_items)
    print(f"  Train: {len(train_items)} filas | índice único: {len(index_items)} encounters")

    print(f"  Construyendo índice E5-small sobre {len(index_items)} preguntas...")
    retriever = E5Retriever([item["question_es"] for item in index_items])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    with out_path.open("w", encoding="utf-8") as f:
        for split_name, items in splits.items():
            print(f"  Recuperando contextos para split '{split_name}' ({len(items)} filas)...")
            for item in items:
                # Leave-one-out solo en train
                exclude = clean_text(item.get("encounter_id", "")) if split_name == "train" else None
                contexts = retrieve_contexts(item, retriever, index_items, top_k, exclude_encounter_id=exclude)

                record = {
                    "split": item["split"],
                    "encounter_id": item["encounter_id"],
                    "image_id": item["image_id"],
                    "image_path": item["image_path"],
                    "question_es": item["question_es"],
                    "answer_es": item["reference_answer_es"],
                    "rag_contexts": contexts,
                    "rag_retriever": "e5_small",
                    "rag_top_k": top_k,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1

            print(f"    -> {len(items)} filas procesadas")

    print(f"\nDataset RAG guardado en {out_path} ({total} filas)")

    # Stats de cobertura
    sample_counts = [0] * (top_k + 1)
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            n = len(json.loads(line).get("rag_contexts", []))
            sample_counts[min(n, top_k)] += 1
    for k, count in enumerate(sample_counts):
        if count:
            print(f"  {count} filas con {k} contextos recuperados")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-computa contextos RAG para entrenamiento RAG-aware")
    p.add_argument(
        "--dataset",
        choices=list(DATASET_CONFIGS),
        required=True,
        help="Dataset a procesar",
    )
    p.add_argument("--dataset-path", type=Path, default=None, help="Ruta alternativa al dataset")
    p.add_argument("--top-k", type=int, default=3, help="Contextos a recuperar por fila (default: 3)")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Ruta de salida del JSONL (default: outputs/datasets/<nombre>_rag_e5_small.jsonl)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = DATASET_CONFIGS[args.dataset]
    out_path = args.out or (DATASETS_DIR / OUT_NAMES[args.dataset])
    build_rag_dataset(config, args.dataset_path, args.top_k, out_path)


if __name__ == "__main__":
    main()
