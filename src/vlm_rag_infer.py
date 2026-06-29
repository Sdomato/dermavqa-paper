"""
Inference VLM with lightweight textual RAG over by-image datasets.

This script does not train anything. It keeps the same VLM input unit used by
the LoRA experiments (one image + one question) and augments the prompt with
top-k similar train cases retrieved by question text.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.vlm_by_image_utils import (
    MODEL_ID,
    SYSTEM_PROMPT,
    ByImageDatasetConfig,
    ImageResolver,
    build_inference_items,
    clean_text,
    filter_split,
    generate_answer,
    load_by_image_dataset,
    load_model_and_processor,
)
from src.vlm_infer_enriched import CONFIG as ENRICHED_CONFIG
from src.vlm_infer_longest_by_image import CONFIG as LONGEST_BY_IMAGE_CONFIG


DATASET_CONFIGS: dict[str, ByImageDatasetConfig] = {
    "enriched": ENRICHED_CONFIG,
    "dataset_enriched": ENRICHED_CONFIG,
    "longest": LONGEST_BY_IMAGE_CONFIG,
    "dataset_longest_answer": LONGEST_BY_IMAGE_CONFIG,
    "dataset_longest_answer_by_image": LONGEST_BY_IMAGE_CONFIG,
}

CONTEXT_LABELS = {
    "dataset_enriched": "enriched",
    "dataset_longest_answer": "longest",
}
E5_SMALL_MODEL_ID = "intfloat/multilingual-e5-small"
SBERT_MINILM_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

RAG_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + " Usa los casos similares recuperados solo como contexto auxiliar. "
    + "No los menciones como fuente, no copies respuestas si no corresponden "
    + "y prioriza la imagen y la consulta actual."
)


def dataset_config(label: str) -> ByImageDatasetConfig:
    try:
        return DATASET_CONFIGS[label]
    except KeyError as exc:
        valid = ", ".join(sorted(DATASET_CONFIGS))
        raise ValueError(f"Dataset no soportado: {label}. Opciones: {valid}") from exc


def unique_by_encounter(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in items:
        encounter_id = clean_text(item.get("encounter_id", ""))
        if not encounter_id or encounter_id in seen:
            continue
        seen.add(encounter_id)
        unique.append(item)
    return unique


class RetrieverIndex(Protocol):
    method: str
    model_name: str

    def search(self, query: str) -> np.ndarray:
        ...


class TfidfRetrieverIndex:
    method = "tfidf"
    model_name = "TF-IDF"

    def __init__(self, train_questions: list[str]) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            norm="l2",
        )
        self.matrix = self.vectorizer.fit_transform(train_questions)

    def search(self, query: str) -> np.ndarray:
        query_matrix = self.vectorizer.transform([query])
        return cosine_similarity(query_matrix, self.matrix)[0]


class SentenceEmbeddingRetrieverIndex:
    def __init__(
        self,
        method: str,
        model_name: str,
        train_questions: list[str],
        query_prefix: str = "",
        corpus_prefix: str = "",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.method = method
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.model = SentenceTransformer(model_name)
        corpus_texts = [f"{corpus_prefix}{question}" for question in train_questions]
        self.matrix = self.model.encode(
            corpus_texts,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=True,
        )

    def search(self, query: str) -> np.ndarray:
        embedding = self.model.encode(
            [f"{self.query_prefix}{query}"],
            normalize_embeddings=True,
            batch_size=1,
            show_progress_bar=False,
        )
        return np.matmul(embedding, self.matrix.T)[0]


def build_retrieval_index(
    config: ByImageDatasetConfig,
    dataset_path: Path | None,
    retriever: str,
) -> tuple[RetrieverIndex, list[dict[str, Any]]]:
    train_records = filter_split(
        load_by_image_dataset(dataset_path, config.default_paths, config.missing_message),
        "train",
        config.split_aliases,
        config.dataset_name,
    )
    train_items = build_inference_items(train_records, config.answer_columns, resolver=ImageResolver())
    train_items = unique_by_encounter(train_items)
    if not train_items:
        raise ValueError(f"No hay items train para construir RAG en {config.dataset_name}")

    train_questions = [item["question_es"] for item in train_items]
    if retriever == "tfidf":
        index: RetrieverIndex = TfidfRetrieverIndex(train_questions)
    elif retriever == "e5_small":
        index = SentenceEmbeddingRetrieverIndex(
            method="e5_small",
            model_name=E5_SMALL_MODEL_ID,
            train_questions=train_questions,
            query_prefix="query: ",
            corpus_prefix="passage: ",
        )
    elif retriever == "sbert_multilingual_minilm":
        index = SentenceEmbeddingRetrieverIndex(
            method="sbert_multilingual_minilm",
            model_name=SBERT_MINILM_MODEL_ID,
            train_questions=train_questions,
        )
    else:
        raise ValueError(f"Retriever no soportado: {retriever}")
    return index, train_items


def safe_method_token(value: str) -> str:
    return (
        clean_text(value)
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
        .lower()
    )


def retrieve_contexts(
    query_item: dict[str, Any],
    retrieval_index: RetrieverIndex,
    train_items: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    scores = retrieval_index.search(query_item["question_es"])
    ranked_indices = scores.argsort()[::-1]

    contexts: list[dict[str, Any]] = []
    query_encounter_id = clean_text(query_item.get("encounter_id", ""))
    for raw_index in ranked_indices:
        index = int(raw_index)
        candidate = train_items[index]
        if clean_text(candidate.get("encounter_id", "")) == query_encounter_id:
            continue
        answer = clean_text(candidate.get("reference_answer_es", ""))
        if not answer:
            continue
        contexts.append(
            {
                "encounter_id": candidate["encounter_id"],
                "question_es": candidate["question_es"],
                "answer_es": answer,
                "score": float(scores[index]),
            }
        )
        if len(contexts) >= top_k:
            break
    return contexts


def format_rag_prompt(question: str, contexts: list[dict[str, Any]]) -> str:
    lines = [
        "Consulta actual:",
        clean_text(question),
        "",
        "Casos similares recuperados del conjunto de entrenamiento:",
    ]
    for index, context in enumerate(contexts, start=1):
        lines.extend(
            [
                f"{index}. Pregunta similar: {context['question_es']}",
                f"   Respuesta del caso similar: {context['answer_es']}",
            ]
        )
    lines.extend(
        [
            "",
            "Responde ahora la consulta actual en espanol clinico, claro y prudente.",
            "No digas que estas usando RAG ni que 'las respuestas sugieren'.",
        ]
    )
    return "\n".join(lines)


def build_rag_messages(
    item: dict[str, Any],
    contexts: list[dict[str, Any]],
    system_prompt: str = RAG_SYSTEM_PROMPT,
) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = [
        {"type": "image", "image": image_path} for image_path in item["image_paths"]
    ]
    user_content.append({"type": "text", "text": format_rag_prompt(item["question_es"], contexts)})
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]


def method_name(
    config: ByImageDatasetConfig,
    context_config: ByImageDatasetConfig,
    adapter: str | None,
    retriever: str,
) -> str:
    base_method = config.lora_method if adapter else config.zero_shot_method
    context_label = CONTEXT_LABELS.get(context_config.dataset_variant, context_config.dataset_variant)
    return f"{base_method}_rag_{safe_method_token(retriever)}_{context_label}"


def run(args: argparse.Namespace) -> None:
    query_config = dataset_config(args.query_dataset)
    context_config = dataset_config(args.context_dataset)

    query_records = filter_split(
        load_by_image_dataset(args.query_dataset_path, query_config.default_paths, query_config.missing_message),
        args.split,
        query_config.split_aliases,
        query_config.dataset_name,
    )
    if args.limit:
        query_records = query_records[: args.limit]

    query_items = build_inference_items(query_records, query_config.answer_columns)
    retrieval_index, train_items = build_retrieval_index(
        context_config, args.context_dataset_path, args.retriever
    )

    method = method_name(query_config, context_config, args.adapter, retrieval_index.method)
    model_name = Path(args.adapter).name if args.adapter else args.model
    out_dir = query_config.results_root / method
    pred_path = out_dir / f"predictions_{args.split}.csv"

    n_missing = sum(1 for item in query_items if item["missing_image_ids"])
    total_missing = sum(len(item["missing_image_ids"]) for item in query_items)
    print(f"Split '{args.split}': {len(query_items)} filas por imagen")
    print(f"  Query dataset: {query_config.dataset_name}")
    print(f"  Context dataset train: {context_config.dataset_name} ({len(train_items)} casos unicos)")
    print(f"  Imagenes faltantes: {total_missing} (en {n_missing} filas)")
    print(
        f"  Metodo: {method} | Modelo: {model_name} | "
        f"retriever={retrieval_index.model_name} | top_k={args.top_k}"
    )

    if args.dry_run:
        print("\n[DRY-RUN] No se carga el modelo. Ejemplo de prompt RAG:")
        if query_items:
            contexts = retrieve_contexts(query_items[0], retrieval_index, train_items, args.top_k)
            example = build_rag_messages(query_items[0], contexts)
            print(json.dumps(example, ensure_ascii=False, indent=2)[:2500])
            print("\n  retrieved_encounter_ids:", [context["encounter_id"] for context in contexts])
            print(f"  reference_answer_es: {query_items[0]['reference_answer_es'][:300]}")
        return

    model, processor = load_model_and_processor(args.model, args.quantize, args.adapter)

    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    for index, item in enumerate(query_items, 1):
        contexts = retrieve_contexts(item, retrieval_index, train_items, args.top_k)
        messages = build_rag_messages(item, contexts)
        start = time.perf_counter()
        prediction = generate_answer(model, processor, messages, args.max_new_tokens)
        latencies.append(time.perf_counter() - start)
        rows.append(
            {
                "split": item["split"],
                "encounter_id": item["encounter_id"],
                "image_id": item["image_id"],
                "image_path": item["image_path"],
                "question_es": item["question_es"],
                "reference_answer_es": item["reference_answer_es"],
                "predicted_answer_es": prediction,
                "model_name": model_name,
                "dataset_variant": query_config.dataset_variant,
                "method": method,
                "rag_context_dataset": context_config.dataset_variant,
                "rag_retriever": retrieval_index.method,
                "rag_retriever_model": retrieval_index.model_name,
                "rag_top_k": args.top_k,
                "retrieved_encounter_ids": json.dumps(
                    [context["encounter_id"] for context in contexts], ensure_ascii=False
                ),
                "retrieved_scores": json.dumps(
                    [round(context["score"], 6) for context in contexts], ensure_ascii=False
                ),
                "rag_context_es": json.dumps(contexts, ensure_ascii=False),
            }
        )
        if index % 10 == 0 or index == len(query_items):
            print(f"  {index}/{len(query_items)} generados")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(pred_path, index=False, encoding="utf-8")
    print(f"\nPredicciones guardadas en {pred_path}")

    runtime = {
        "method": method,
        "model_name": model_name,
        "dataset_variant": query_config.dataset_variant,
        "rag_context_dataset": context_config.dataset_variant,
        "rag_retriever": retrieval_index.method,
        "rag_retriever_model": retrieval_index.model_name,
        "rag_top_k": args.top_k,
        "split": args.split,
        "n": len(rows),
        "mean_latency_s": statistics.mean(latencies) if latencies else 0.0,
        "total_time_s": sum(latencies),
        "max_new_tokens": args.max_new_tokens,
        "quantize": args.quantize,
    }
    with (out_dir / f"runtime_{args.split}.json").open("w", encoding="utf-8") as handle:
        json.dump(runtime, handle, ensure_ascii=False, indent=2)
    print(f"Metricas operativas: {runtime['mean_latency_s']:.2f}s/ejemplo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inferencia VLM con RAG textual train-only")
    parser.add_argument("--query-dataset", choices=sorted(DATASET_CONFIGS), required=True)
    parser.add_argument("--context-dataset", choices=sorted(DATASET_CONFIGS), required=True)
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--query-dataset-path", type=Path, default=None)
    parser.add_argument("--context-dataset-path", type=Path, default=None)
    parser.add_argument("--model", default=MODEL_ID, help=f"Modelo HF (default: {MODEL_ID})")
    parser.add_argument("--adapter", default=None, help="Ruta a adapter LoRA")
    parser.add_argument("--quantize", choices=["4bit", "8bit"], default="4bit")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--retriever",
        choices=["tfidf", "e5_small", "sbert_multilingual_minilm"],
        default="e5_small",
        help="Retriever textual para construir el contexto RAG",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
