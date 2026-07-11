"""
Held-out textual retrieval baselines for the paper tables.

This script evaluates retrieval using only the train split as the retrieval
index, then queries valid/test cases. It is CPU-friendly and implements
TF-IDF, multilingual E5-small and multilingual SBERT, giving leakage-free
baselines for the longest-answer, short-answer and longest-by-image dataset
variants.

Outputs:
  outputs/results/<dataset_variant>/retrieval_textual_heldout/
    predictions_valid_<method>.csv
    predictions_test_<method>.csv
  outputs/metrics/<dataset_variant>/retrieval_heldout/
    metrics_summary.csv
    metrics_per_case.csv

Usage:
  python -m src.evaluate_retrieval_heldout --dataset all
  python -m src.evaluate_retrieval_heldout --dataset dataset_longest_answer_by_image --methods tfidf,e5,sbert
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import sacrebleu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

METHOD_LABELS = {
    "tfidf": ("retrieval_tfidf_train_only", "TF-IDF"),
    "e5": ("retrieval_e5_train_only", "intfloat/multilingual-e5-small"),
    "sbert": ("retrieval_sbert_train_only", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
}


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "dataset_longest_answer": {
        "path": PROJECT_ROOT / "outputs" / "datasets" / "dataset_longest_answer.csv",
        "target": "longest_answer",
        "split_col": "_split",
        "splits": {"valid_ht": "valid", "test_ht_spanishtestsetcorrected": "test"},
    },
    "dataset_short_answer": {
        "path": PROJECT_ROOT / "outputs" / "datasets" / "dataset_short_answer.csv",
        "target": "short_answer",
        "split_col": "_split",
        "splits": {"valid_ht": "valid", "test_ht_spanishtestsetcorrected": "test"},
    },
    "dataset_longest_answer_by_image": {
        "path": PROJECT_ROOT / "outputs" / "datasets" / "dataset_longest_answer_by_image.csv",
        "target": "longest_answer",
        "split_col": "split",
        "splits": {"valid": "valid", "test": "test"},
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def question_text(record: dict[str, str]) -> str:
    if "question_es" in record:
        return clean_text(record.get("question_es", ""))
    return clean_text(f"{record.get('query_title_es', '')} {record.get('query_content_es', '')}")


def normalize(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize(prediction).split()
    ref_tokens = normalize(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize(prediction).split()
    ref_tokens = normalize(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = lcs_length(pred_tokens, ref_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_predictions(
    dataset_variant: str,
    target: str,
    split: str,
    predictions: list[dict[str, Any]],
    elapsed_s: float,
    method_key: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    method_name, model_name = METHOD_LABELS[method_key]
    pred_texts = [clean_text(row["predicted_answer_es"]) for row in predictions]
    ref_texts = [clean_text(row["reference_answer_es"]) for row in predictions]
    rouge_values = [rouge_l_f1(pred, ref) for pred, ref in zip(pred_texts, ref_texts)]
    token_values = [token_f1(pred, ref) for pred, ref in zip(pred_texts, ref_texts)]
    chrf_values = [
        sacrebleu.corpus_chrf([pred], [[ref]]).score / 100.0
        for pred, ref in zip(pred_texts, ref_texts)
    ]
    corpus_bleu = sacrebleu.corpus_bleu(pred_texts, [ref_texts]).score
    corpus_chrf = sacrebleu.corpus_chrf(pred_texts, [ref_texts]).score

    per_case: list[dict[str, Any]] = []
    for row, rouge_l, tok_f1, chrf in zip(predictions, rouge_values, token_values, chrf_values):
        per_case.append(
            {
                "dataset_variant": dataset_variant,
                "target": target,
                "method": method_name,
                "model": model_name,
                "split": split,
                "unit": "case",
                "encounter_id": row["encounter_id"],
                "retrieved_encounter_id": row["retrieved_encounter_id"],
                "similarity_score": row["similarity_score"],
                "rouge_l": rouge_l,
                "chrf": chrf,
                "token_f1": tok_f1,
            }
        )

    summary = {
        "dataset_variant": dataset_variant,
        "target": target,
        "method": method_name,
        "model": model_name,
        "split": split,
        "unit": "case",
        "n": len(predictions),
        "sacrebleu": corpus_bleu,
        "chrf_corpus": corpus_chrf,
        "chrf_mean": mean(chrf_values),
        "rouge_l_mean": mean(rouge_values),
        "token_f1_mean": mean(token_values),
        "mean_latency_s": elapsed_s / len(predictions) if predictions else 0.0,
        "source": f"outputs/results/{dataset_variant}/retrieval_textual_heldout/predictions_{split}_{method_key}.csv",
        "notes": f"{model_name} retrieval with train split as the only retrieval index",
    }
    return per_case, summary


def encode_e5(train_texts: list[str], query_texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(1)
    model_id = "intfloat/multilingual-e5-small"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()

    def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def encode(texts: list[str], prefix: str) -> np.ndarray:
        all_embeddings: list[np.ndarray] = []
        for start in range(0, len(texts), 64):
            batch = [prefix + t for t in texts[start : start + 64]]
            encoded = tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                output = model(**encoded)
            embeddings = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings)

    train_emb = encode(train_texts, "passage: ")
    query_emb = encode(query_texts, "query: ")
    return train_emb, query_emb


def encode_sbert(train_texts: list[str], query_texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(1)
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    train_emb = model.encode(train_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    query_emb = model.encode(query_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(train_emb), np.asarray(query_emb)


def compute_similarities(
    method_key: str, train_texts: list[str], query_texts: list[str]
) -> np.ndarray:
    """Returns a (n_queries, n_train) similarity matrix."""
    if method_key == "tfidf":
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            norm="l2",
        )
        train_matrix = vectorizer.fit_transform(train_texts)
        query_matrix = vectorizer.transform(query_texts)
        return cosine_similarity(query_matrix, train_matrix)
    if method_key == "e5":
        train_emb, query_emb = encode_e5(train_texts, query_texts)
    elif method_key == "sbert":
        train_emb, query_emb = encode_sbert(train_texts, query_texts)
    else:
        raise ValueError(f"Unknown method: {method_key}")
    return query_emb @ train_emb.T


def run_dataset(dataset_variant: str, methods: list[str]) -> None:
    config = DATASETS[dataset_variant]
    split_col = config["split_col"]
    splits = config["splits"]
    records = read_csv(config["path"])
    train_records = [row for row in records if row.get(split_col) == "train"]
    if not train_records:
        raise ValueError(f"No train records found for {dataset_variant}")
    train_texts = [question_text(row) for row in train_records]

    predictions_dir = PROJECT_ROOT / "outputs" / "results" / dataset_variant / "retrieval_textual_heldout"
    metrics_dir = PROJECT_ROOT / "outputs" / "metrics" / dataset_variant / "retrieval_heldout"
    all_per_case: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for method_key in methods:
        method_name, model_name = METHOD_LABELS[method_key]
        for raw_split, split in splits.items():
            query_records = [row for row in records if row.get(split_col) == raw_split]
            query_texts = [question_text(row) for row in query_records]
            start = time.perf_counter()
            similarities = compute_similarities(method_key, train_texts, query_texts)
            elapsed_s = time.perf_counter() - start

            predictions: list[dict[str, Any]] = []
            for row_index, query_record in enumerate(query_records):
                best_index = int(similarities[row_index].argmax())
                retrieved = train_records[best_index]
                predictions.append(
                    {
                        "dataset_variant": dataset_variant,
                        "target": config["target"],
                        "method": method_name,
                        "model": model_name,
                        "split": split,
                        "unit": "case",
                        "encounter_id": query_record["encounter_id"],
                        "retrieved_encounter_id": retrieved["encounter_id"],
                        "similarity_score": round(float(similarities[row_index, best_index]), 6),
                        "question_es": question_text(query_record),
                        "reference_answer_es": clean_text(query_record.get("answer_es", "")),
                        "predicted_answer_es": clean_text(retrieved.get("answer_es", "")),
                    }
                )

            write_csv(
                predictions_dir / f"predictions_{split}_{method_key}.csv",
                predictions,
                [
                    "dataset_variant",
                    "target",
                    "method",
                    "model",
                    "split",
                    "unit",
                    "encounter_id",
                    "retrieved_encounter_id",
                    "similarity_score",
                    "question_es",
                    "reference_answer_es",
                    "predicted_answer_es",
                ],
            )
            per_case, summary = evaluate_predictions(
                dataset_variant=dataset_variant,
                target=config["target"],
                split=split,
                predictions=predictions,
                elapsed_s=elapsed_s,
                method_key=method_key,
            )
            all_per_case.extend(per_case)
            summaries.append(summary)
            print(
                f"{dataset_variant} [{method_key}] {split}: {len(predictions)} cases, "
                f"chrF={summary['chrf_mean']:.3f}"
            )

    write_csv(
        metrics_dir / "metrics_per_case.csv",
        all_per_case,
        [
            "dataset_variant",
            "target",
            "method",
            "model",
            "split",
            "unit",
            "encounter_id",
            "retrieved_encounter_id",
            "similarity_score",
            "rouge_l",
            "chrf",
            "token_f1",
        ],
    )
    write_csv(
        metrics_dir / "metrics_summary.csv",
        summaries,
        [
            "dataset_variant",
            "target",
            "method",
            "model",
            "split",
            "unit",
            "n",
            "sacrebleu",
            "chrf_corpus",
            "chrf_mean",
            "rouge_l_mean",
            "token_f1_mean",
            "mean_latency_s",
            "source",
            "notes",
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate leakage-free held-out retrieval baselines.")
    parser.add_argument(
        "--dataset",
        choices=["all", *DATASETS.keys()],
        default="all",
        help="Dataset variant to evaluate.",
    )
    parser.add_argument(
        "--methods",
        default="tfidf",
        help="Comma-separated methods to run: tfidf,e5,sbert",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    variants = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}
    for dataset_variant in variants:
        run_dataset(dataset_variant, methods)


if __name__ == "__main__":
    main()
