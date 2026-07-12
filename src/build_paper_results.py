"""
Build paper-ready result tables and figures from existing DermaVQA artifacts.

This script intentionally avoids heavy ML dependencies so it can run on the
local Windows checkout. If `sacrebleu` is installed, it also computes missing
corpus sacreBLEU/chrF scores from prediction CSVs. It does not recompute
BERTScore because that requires loading a transformer model.

Outputs:
  outputs/paper/tables/
    paper_all_metrics_long.csv
    paper_main_test_comparison.csv
    paper_dataset_split_counts.csv
    paper_missing_metrics_report.md
    paper_results_summary.md
  outputs/paper/figures/
    dataset_split_counts.svg
    main_test_chrf.svg
    main_test_bertscore.svg
    main_test_rouge_tokenf1.svg
    vlm_latency_vs_bertscore.svg
    README.md
"""

from __future__ import annotations

import ast
import csv
import html
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_TABLES_DIR = PROJECT_ROOT / "outputs" / "paper" / "tables"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "paper" / "figures"
OUTPUT_PAPER_DIR = PROJECT_ROOT / "outputs" / "paper"


METRIC_COLUMNS = [
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
    "bertscore_f1_mean",
    "mean_latency_s",
    "source",
    "notes",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_first(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    return rows[0] if rows else {}


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").lower().split())


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    common = sum((pred_counts & ref_counts).values())
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
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = lcs_length(pred_tokens, ref_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def lexical_from_predictions(path: Path) -> dict[str, float]:
    rows = read_csv(path)
    if not rows:
        return {}
    rouge_values: list[float] = []
    token_values: list[float] = []
    chrf_values: list[float] = []
    empty_predictions = 0
    try:
        import sacrebleu
    except ImportError:
        sacrebleu = None
    for row in rows:
        prediction = row.get("predicted_answer_es", "")
        reference = row.get("reference_answer_es", "")
        if not prediction.strip():
            empty_predictions += 1
        rouge_values.append(rouge_l_f1(prediction, reference))
        token_values.append(token_f1(prediction, reference))
        if sacrebleu is not None:
            chrf_values.append(sacrebleu.corpus_chrf([prediction], [[reference]]).score / 100.0)
    metrics = {
        "rouge_l_mean": sum(rouge_values) / len(rouge_values),
        "token_f1_mean": sum(token_values) / len(token_values),
        "empty_predictions": float(empty_predictions),
    }
    if chrf_values:
        metrics["chrf_mean"] = sum(chrf_values) / len(chrf_values)
    return metrics


def corpus_scores(predictions: list[str], references: list[str]) -> dict[str, float]:
    if not predictions or not references:
        return {}
    try:
        import sacrebleu
    except ImportError:
        return {}
    return {
        "sacrebleu": sacrebleu.corpus_bleu(predictions, [references]).score,
        "chrf_corpus": sacrebleu.corpus_chrf(predictions, [references]).score,
    }


def corpus_scores_from_prediction_csv(path: Path) -> dict[str, float]:
    rows = read_csv(path)
    predictions = [row.get("predicted_answer_es", "") for row in rows]
    references = [row.get("reference_answer_es", "") for row in rows]
    return corpus_scores(predictions, references)


def case_concat_corpus_scores_from_prediction_csv(path: Path) -> tuple[dict[str, float], int]:
    rows = read_csv(path)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        encounter_id = row.get("encounter_id", "")
        if encounter_id:
            grouped[encounter_id].append(row)
    predictions: list[str] = []
    references: list[str] = []
    for encounter_id in sorted(grouped):
        case_rows = sorted(grouped[encounter_id], key=lambda row: row.get("image_id", ""))
        reference = case_rows[0].get("reference_answer_es", "")
        seen_predictions: set[str] = set()
        unique_predictions: list[str] = []
        for row in case_rows:
            prediction = " ".join(row.get("predicted_answer_es", "").split())
            key = normalize_text(prediction)
            if prediction and key not in seen_predictions:
                seen_predictions.add(key)
                unique_predictions.append(prediction)
        predictions.append(" ".join(unique_predictions))
        references.append(reference)
    return corpus_scores(predictions, references), len(predictions)


def metric_row(**kwargs: Any) -> dict[str, str]:
    row = {key: "" for key in METRIC_COLUMNS}
    for key, value in kwargs.items():
        if key in {"n"}:
            row[key] = str(value)
        elif isinstance(value, float):
            row[key] = fmt(value)
        else:
            row[key] = "" if value is None else str(value)
    return row


def method_label(method: str) -> str:
    labels = {
        "retrieval_tfidf": "TF-IDF",
        "retrieval_e5": "E5",
        "retrieval_sbert": "SBERT",
        "retrieval_textual_tfidf": "TF-IDF",
        "retrieval_textual_e5": "E5",
        "retrieval_textual_sbert": "SBERT",
        "retrieval_tfidf_train_only": "TF-IDF train-only",
        "retrieval_visual": "Visual",
        "retrieval_multimodal": "Multimodal",
        "vlm_zero_shot": "VLM zero-shot",
        "vlm_zero_shot_by_image": "VLM zero-shot by-image",
        "vlm_zero_shot_rag_e5_small_enriched": "VLM zero-shot + RAG",
        "vlm_zero_shot_by_image_rag_e5_small_longest": "VLM zero-shot + RAG by-image",
        "vlm_lora": "VLM LoRA",
        "vlm_lora_case_avg": "VLM LoRA case-avg",
        "vlm_lora_rag_e5_small_enriched": "VLM LoRA + RAG",
        "vlm_lora_by_image": "VLM LoRA by-image",
        "vlm_lora_by_image_rag_e5_small_longest": "VLM LoRA + RAG by-image",
    }
    return labels.get(method, method)


def split_label(split: str) -> str:
    return {
        "valid_ht": "valid",
        "test_ht_spanishtestsetcorrected": "test",
    }.get(split, split)


def build_enriched_retrieval_rows() -> list[dict[str, str]]:
    metrics_path = PROJECT_ROOT / "outputs" / "metrics" / "dataset_enriched" / "retrieval_textual" / "metrics_summary.csv"
    bert_path = PROJECT_ROOT / "outputs" / "metrics" / "dataset_enriched" / "retrieval_textual" / "bertscore_summary.csv"
    bert_lookup = {
        (row.get("method", ""), row.get("split", "")): safe_float(row.get("bertscore_f1"))
        for row in read_csv(bert_path)
    }
    model_lookup = {
        "tfidf": "TF-IDF",
        "e5_small": "intfloat/multilingual-e5-small",
        "sbert_multilingual_minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    }
    method_lookup = {
        "tfidf": "retrieval_tfidf",
        "e5_small": "retrieval_e5",
        "sbert_multilingual_minilm": "retrieval_sbert",
    }
    rows: list[dict[str, str]] = []
    for source in read_csv(metrics_path):
        raw_method = source.get("method", "")
        split = source.get("split", "")
        predictions_path = (
            PROJECT_ROOT
            / "results"
            / "dataset_enriched"
            / "retrieval_textual"
            / f"predictions_{split}_{raw_method}.csv"
        )
        lexical = lexical_from_predictions(predictions_path)
        rows.append(
            metric_row(
                dataset_variant="dataset_enriched",
                target="enriched_answer",
                method=method_lookup.get(raw_method, raw_method),
                model=model_lookup.get(raw_method, raw_method),
                split=split,
                unit="case",
                n=source.get("rows", ""),
                sacrebleu=safe_float(source.get("sacrebleu")),
                chrf_corpus=safe_float(source.get("chrf")),
                chrf_mean=lexical.get("chrf_mean"),
                rouge_l_mean=lexical.get("rouge_l_mean"),
                token_f1_mean=lexical.get("token_f1_mean"),
                bertscore_f1_mean=bert_lookup.get((raw_method, split)),
                source="outputs/metrics/dataset_enriched/retrieval_textual",
                notes="retrieval textual enriched; ROUGE-L/token-F1 recomputed from legacy prediction CSVs",
            )
        )
    return rows


def build_enriched_vlm_rows() -> list[dict[str, str]]:
    metrics_path = PROJECT_ROOT / "outputs" / "metrics" / "dataset_enriched" / "metrics_mixed.csv"
    rows: list[dict[str, str]] = []
    for source in read_csv(metrics_path):
        split = source.get("split", "")
        method = source.get("method", "")
        runtime = runtime_for("dataset_enriched", method, split)
        model = source.get("model_name", "")
        if model == "final_adapter":
            model = "Qwen/Qwen2.5-VL-3B-Instruct + LoRA"
        rows.append(
            metric_row(
                dataset_variant="dataset_enriched",
                target="enriched_answer",
                method=method,
                model=model,
                split=split,
                unit="image",
                n=source.get("n", ""),
                sacrebleu=safe_float(source.get("sacrebleu_corpus")),
                chrf_corpus=safe_float(source.get("chrf_corpus")),
                chrf_mean=safe_float(source.get("chrf_mean")),
                rouge_l_mean=safe_float(source.get("rouge_l_mean")),
                token_f1_mean=safe_float(source.get("token_f1_mean")),
                bertscore_f1_mean=safe_float(source.get("bertscore_f1_mean")),
                mean_latency_s=safe_float(runtime.get("mean_latency_s")),
                source="outputs/metrics/dataset_enriched/metrics_mixed.csv",
                notes="VLM enriched evaluated per image; RAG uses E5 retrieval when method name includes rag",
            )
        )
    return rows


def build_enriched_vlm_case_average_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    metrics_dir = PROJECT_ROOT / "outputs" / "metrics" / "dataset_enriched"
    runtime_dir = PROJECT_ROOT / "outputs" / "results" / "dataset_enriched" / "vlm_lora"
    for split, filename in [
        ("valid", "per_case_vlm_lora_valid.csv"),
        ("test", "per_case_vlm_lora_test.csv"),
    ]:
        source_rows = read_csv(metrics_dir / filename)
        grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in source_rows:
            encounter_id = row.get("encounter_id", "")
            if not encounter_id:
                continue
            for metric in ["chrf", "rouge_l", "token_f1", "bertscore_f1"]:
                value = safe_float(row.get(metric))
                if value is not None:
                    grouped[encounter_id][metric].append(value)
        if not grouped:
            continue
        case_metrics: dict[str, list[float]] = defaultdict(list)
        for metric_lists in grouped.values():
            for metric, values in metric_lists.items():
                if values:
                    case_metrics[metric].append(sum(values) / len(values))
        runtime = read_json_flat(runtime_dir / f"runtime_{split}.json")
        total_time = safe_float(runtime.get("total_time_s"))
        case_count = len(grouped)
        corpus, corpus_count = case_concat_corpus_scores_from_prediction_csv(
            runtime_dir / f"predictions_{split}.csv"
        )
        rows.append(
            metric_row(
                dataset_variant="dataset_enriched",
                target="enriched_answer",
                method="vlm_lora_case_avg",
                model="Qwen/Qwen2.5-VL-3B-Instruct",
                split=split,
                unit="case",
                n=case_count,
                sacrebleu=corpus.get("sacrebleu"),
                chrf_corpus=corpus.get("chrf_corpus"),
                chrf_mean=sum(case_metrics["chrf"]) / len(case_metrics["chrf"]) if case_metrics["chrf"] else None,
                rouge_l_mean=sum(case_metrics["rouge_l"]) / len(case_metrics["rouge_l"]) if case_metrics["rouge_l"] else None,
                token_f1_mean=sum(case_metrics["token_f1"]) / len(case_metrics["token_f1"]) if case_metrics["token_f1"] else None,
                bertscore_f1_mean=sum(case_metrics["bertscore_f1"]) / len(case_metrics["bertscore_f1"]) if case_metrics["bertscore_f1"] else None,
                mean_latency_s=(total_time / case_count) if total_time and case_count else None,
                source=f"outputs/metrics/dataset_enriched/{filename}",
                notes=(
                    "case-level average over image-level metrics; corpus scores use "
                    f"deduplicated image predictions concatenated per encounter (n={corpus_count})"
                ),
            )
        )
    return rows


def read_json_flat(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    import json

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return {key: str(value) for key, value in payload.items() if not isinstance(value, (dict, list))}


def build_longest_retrieval_rows() -> list[dict[str, str]]:
    metrics_path = PROJECT_ROOT / "outputs" / "metrics" / "dataset_longest_answer" / "metrics_summary.csv"
    mapping = [
        ("retrieval_textual/tfidf_results", "retrieval_tfidf", "TF-IDF"),
        ("retrieval_textual_e5/e5_results", "retrieval_e5", "intfloat/multilingual-e5-base"),
        ("retrieval_textual_sbert/sbert_results", "retrieval_sbert", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        ("retrieval_visual/visual_results", "retrieval_visual", "BiomedCLIP"),
        ("retrieval_multimodal/multimodal_alpha0.60_results", "retrieval_multimodal", "alpha0.60"),
    ]
    lookup = {source: (method, model) for source, method, model in mapping}
    rows: list[dict[str, str]] = []
    for source in read_csv(metrics_path):
        method, model = lookup.get(source.get("model", ""), (source.get("model", ""), source.get("model", "")))
        rows.append(
            metric_row(
                dataset_variant="dataset_longest_answer",
                target="longest_answer",
                method=method,
                model=model,
                split=split_label(source.get("split", "all") or "all"),
                unit="case",
                n=source.get("n", ""),
                chrf_mean=safe_float(source.get("chrf_mean")),
                rouge_l_mean=safe_float(source.get("rouge_l_mean")),
                token_f1_mean=safe_float(source.get("token_f1_mean")),
                bertscore_f1_mean=safe_float(source.get("bertscore_f1_mean")),
                source="outputs/metrics/dataset_longest_answer/metrics_summary.csv",
                notes="retrieval summary from all-index results; split column is evaluation subset, not train-only retrieval",
            )
        )
    return rows


def build_retrieval_heldout_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset_variant, target in [
        ("dataset_longest_answer", "longest_answer"),
        ("dataset_enriched", "enriched_answer"),
    ]:
        metrics_path = (
            PROJECT_ROOT
            / "outputs"
            / "metrics"
            / dataset_variant
            / "retrieval_heldout"
            / "metrics_summary.csv"
        )
        for source in read_csv(metrics_path):
            rows.append(
                metric_row(
                    dataset_variant=dataset_variant,
                    target=source.get("target", target),
                    method=source.get("method", "retrieval_tfidf_train_only"),
                    model=source.get("model", "TF-IDF"),
                    split=source.get("split", ""),
                    unit=source.get("unit", "case"),
                    n=source.get("n", ""),
                    sacrebleu=safe_float(source.get("sacrebleu")),
                    chrf_corpus=safe_float(source.get("chrf_corpus")),
                    chrf_mean=safe_float(source.get("chrf_mean")),
                    rouge_l_mean=safe_float(source.get("rouge_l_mean")),
                    token_f1_mean=safe_float(source.get("token_f1_mean")),
                    mean_latency_s=safe_float(source.get("mean_latency_s")),
                    source=str(metrics_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    notes=source.get("notes", "held-out retrieval with train-only retrieval index"),
                )
            )
    return rows


def aggregate_per_case_metrics(path: Path) -> dict[str, float]:
    rows = read_csv(path)
    if not rows:
        return {}
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key in ["chrf", "rouge_l", "token_f1", "bertscore_f1"]:
            value = safe_float(row.get(key))
            if value is not None:
                values[key].append(value)
    return {key: sum(items) / len(items) for key, items in values.items() if items}


def runtime_for(dataset: str, method: str, split: str) -> dict[str, str]:
    path = PROJECT_ROOT / "outputs" / "results" / dataset / method / f"runtime_{split}.json"
    return read_json_flat(path)


def build_longest_vlm_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    metrics_mixed = PROJECT_ROOT / "outputs" / "metrics" / "dataset_longest_answer" / "metrics_mixed.csv"
    for source in read_csv(metrics_mixed):
        split = source.get("split", "")
        simple_split = split_label(split)
        method = source.get("method", "")
        runtime = runtime_for("dataset_longest_answer", method, simple_split)
        model = source.get("model_name", "")
        if model == "final_adapter":
            model = "Qwen/Qwen2.5-VL-3B-Instruct + LoRA"
        rows.append(
            metric_row(
                dataset_variant="dataset_longest_answer",
                target="longest_answer",
                method=method,
                model=model,
                split=simple_split,
                unit="image" if "by_image" in method else "case",
                n=source.get("n", ""),
                sacrebleu=safe_float(source.get("sacrebleu_corpus")),
                chrf_corpus=safe_float(source.get("chrf_corpus")),
                chrf_mean=safe_float(source.get("chrf_mean")),
                rouge_l_mean=safe_float(source.get("rouge_l_mean")),
                token_f1_mean=safe_float(source.get("token_f1_mean")),
                bertscore_f1_mean=safe_float(source.get("bertscore_f1_mean")),
                mean_latency_s=safe_float(runtime.get("mean_latency_s")),
                source="outputs/metrics/dataset_longest_answer/metrics_mixed.csv",
                notes="VLM longest-answer evaluated per image; RAG uses E5 retrieval when method name includes rag",
            )
        )

    per_case_dir = PROJECT_ROOT / "outputs" / "metrics" / "dataset_longest_answer"
    for split, filename in [
        ("valid", "per_case_vlm_lora_valid_ht.csv"),
        ("test", "per_case_vlm_lora_test_ht_spanishtestsetcorrected.csv"),
    ]:
        metrics = aggregate_per_case_metrics(per_case_dir / filename)
        count = len(read_csv(per_case_dir / filename))
        runtime = runtime_for("dataset_longest_answer", "vlm_lora", split)
        corpus = corpus_scores_from_prediction_csv(
            PROJECT_ROOT / "outputs" / "results" / "dataset_longest_answer" / "vlm_lora" / f"predictions_{split}.csv"
        )
        rows.append(
            metric_row(
                dataset_variant="dataset_longest_answer",
                target="longest_answer",
                method="vlm_lora",
                model="Qwen/Qwen2.5-VL-3B-Instruct",
                split=split,
                unit="case",
                n=count,
                sacrebleu=corpus.get("sacrebleu"),
                chrf_corpus=corpus.get("chrf_corpus"),
                chrf_mean=metrics.get("chrf"),
                rouge_l_mean=metrics.get("rouge_l"),
                token_f1_mean=metrics.get("token_f1"),
                bertscore_f1_mean=metrics.get("bertscore_f1"),
                mean_latency_s=safe_float(runtime.get("mean_latency_s")),
                source=f"outputs/metrics/dataset_longest_answer/{filename}",
                notes="VLM LoRA longest per-case metrics; corpus scores recomputed from prediction CSV",
            )
        )
    return rows


def parse_image_count(value: str) -> int:
    try:
        parsed = ast.literal_eval(value)
    except Exception:
        return len([part for part in value.split(";") if part.strip()])
    if isinstance(parsed, list):
        return len(parsed)
    return 0


def build_dataset_split_counts() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset_name, path, split_col in [
        ("dataset_longest_answer", PROJECT_ROOT / "outputs" / "datasets" / "dataset_longest_answer.csv", "_split"),
    ]:
        source_rows = read_csv(path)
        case_counts = Counter(row.get(split_col, "") for row in source_rows)
        image_counts: Counter[str] = Counter()
        for row in source_rows:
            image_counts[row.get(split_col, "")] += parse_image_count(row.get("image_ids", ""))
        for split in ["train", "valid_ht", "test_ht_spanishtestsetcorrected"]:
            rows.append(
                {
                    "dataset_variant": dataset_name,
                    "split": split_label(split),
                    "case_count": str(case_counts.get(split, 0)),
                    "image_row_count": str(image_counts.get(split, 0)),
                }
            )

    enriched_rows = read_csv(PROJECT_ROOT / "outputs" / "datasets" / "dermavqa_iiyi_llm_synthesized_answer_finetune.csv")
    enriched_image_counts = Counter(row.get("split", "") for row in enriched_rows)
    enriched_cases: dict[str, set[str]] = defaultdict(set)
    for row in enriched_rows:
        enriched_cases[row.get("split", "")].add(row.get("encounter_id", ""))
    for split in ["train", "valid", "test"]:
        rows.append(
            {
                "dataset_variant": "dataset_enriched",
                "split": split,
                "case_count": str(len(enriched_cases.get(split, set()))),
                "image_row_count": str(enriched_image_counts.get(split, 0)),
            }
        )
    return rows


def score_for_plot(row: dict[str, str], metric: str) -> float | None:
    if metric == "chrf":
        mean = safe_float(row.get("chrf_mean"))
        if mean is not None:
            return mean
        corpus = safe_float(row.get("chrf_corpus"))
        if corpus is not None:
            return corpus / 100.0
    return safe_float(row.get(metric))


def main_test_rows(all_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    datasets_with_heldout_retrieval = {
        row.get("dataset_variant", "")
        for row in all_rows
        if row.get("split") == "test" and row.get("method") == "retrieval_tfidf_train_only"
    }
    keep = []
    for row in all_rows:
        if row.get("split") != "test":
            continue
        if row.get("dataset_variant") == "dataset_enriched" and row.get("method") == "vlm_lora":
            continue
        if (
            row.get("dataset_variant") in datasets_with_heldout_retrieval
            and row.get("method", "").startswith("retrieval_")
            and row.get("method") != "retrieval_tfidf_train_only"
        ):
            continue
        keep.append(row)

    extra = []
    for row in all_rows:
        if row.get("split") != "all":
            continue
        if row.get("dataset_variant") == "dataset_longest_answer":
            if "dataset_longest_answer" in datasets_with_heldout_retrieval:
                continue
            if row.get("method") in {"retrieval_sbert", "retrieval_multimodal"}:
                extra.append(row)
    order = {
        ("dataset_enriched", "retrieval_tfidf"): 10,
        ("dataset_enriched", "retrieval_e5"): 11,
        ("dataset_enriched", "retrieval_sbert"): 12,
        ("dataset_enriched", "vlm_lora_case_avg"): 13,
        ("dataset_enriched", "vlm_zero_shot"): 14,
        ("dataset_enriched", "vlm_zero_shot_rag_e5_small_enriched"): 15,
        ("dataset_enriched", "vlm_lora_rag_e5_small_enriched"): 16,
        ("dataset_longest_answer", "retrieval_tfidf_train_only"): 20,
        ("dataset_longest_answer", "retrieval_sbert"): 21,
        ("dataset_longest_answer", "retrieval_multimodal"): 22,
        ("dataset_longest_answer", "vlm_zero_shot"): 23,
        ("dataset_longest_answer", "vlm_lora"): 24,
        ("dataset_longest_answer", "vlm_zero_shot_by_image"): 25,
        ("dataset_longest_answer", "vlm_zero_shot_by_image_rag_e5_small_longest"): 26,
        ("dataset_longest_answer", "vlm_lora_by_image"): 27,
        ("dataset_longest_answer", "vlm_lora_by_image_rag_e5_small_longest"): 28,
    }
    combined = keep + extra
    return sorted(
        combined,
        key=lambda row: order.get((row.get("dataset_variant", ""), row.get("method", "")), 999),
    )


def short_name(row: dict[str, str]) -> str:
    dataset_labels = {
        "dataset_enriched": "Enriched",
        "dataset_longest_answer": "Longest",
    }
    dataset = dataset_labels.get(row.get("dataset_variant", ""), row.get("dataset_variant", "dataset"))
    method = method_label(row.get("method", ""))
    if row.get("method") == "retrieval_multimodal":
        method = "MM retrieval"
    return f"{dataset}\n{method}"


def svg_escape(text: Any) -> str:
    return html.escape(str(text), quote=True)


def write_svg_bar_chart(
    path: Path,
    title: str,
    rows: list[dict[str, str]],
    metric: str,
    ylabel: str,
    max_value: float | None = None,
) -> None:
    values = [(short_name(row), score_for_plot(row, metric)) for row in rows]
    values = [(label, value) for label, value in values if value is not None]
    if not values:
        return
    max_score = max(value for _, value in values)
    upper = max_value or max(0.05, max_score * 1.18)
    width = max(900, 95 * len(values) + 140)
    height = 520
    margin_left = 80
    margin_right = 40
    margin_top = 70
    margin_bottom = 140
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    bar_gap = 18
    bar_w = max(28, (plot_w - bar_gap * (len(values) - 1)) / len(values))

    lines = svg_header(width, height, title)
    lines.append(f'<text x="{width/2:.1f}" y="32" text-anchor="middle" class="title">{svg_escape(title)}</text>')
    lines.append(f'<text x="22" y="{margin_top + plot_h/2:.1f}" transform="rotate(-90 22 {margin_top + plot_h/2:.1f})" text-anchor="middle" class="axis-label">{svg_escape(ylabel)}</text>')
    lines.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" class="axis"/>')
    lines.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" class="axis"/>')

    for tick in range(0, 6):
        value = upper * tick / 5
        y = margin_top + plot_h - (value / upper) * plot_h
        lines.append(f'<line x1="{margin_left-5}" y1="{y:.1f}" x2="{margin_left + plot_w}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{margin_left-10}" y="{y+4:.1f}" text-anchor="end" class="tick">{value:.2f}</text>')

    palette = ["#4C78A8", "#72B7B2", "#54A24B", "#E45756", "#F58518", "#B279A2", "#FF9DA6", "#9D755D"]
    for index, (label, value) in enumerate(values):
        x = margin_left + index * (bar_w + bar_gap)
        bar_h = (value / upper) * plot_h
        y = margin_top + plot_h - bar_h
        color = palette[index % len(palette)]
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="3"/>')
        lines.append(f'<text x="{x + bar_w/2:.1f}" y="{y - 8:.1f}" text-anchor="middle" class="value">{value:.3f}</text>')
        for line_index, part in enumerate(label.split("\n")):
            lines.append(f'<text x="{x + bar_w/2:.1f}" y="{margin_top + plot_h + 28 + 16*line_index:.1f}" text-anchor="middle" class="label">{svg_escape(part)}</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_svg_grouped_bar_chart(path: Path, title: str, rows: list[dict[str, str]]) -> None:
    metrics = [("rouge_l_mean", "ROUGE-L", "#4C78A8"), ("token_f1_mean", "Token F1", "#F58518")]
    plot_rows = []
    for row in rows:
        values = [safe_float(row.get(metric)) for metric, _, _ in metrics]
        if any(value is not None for value in values):
            plot_rows.append((short_name(row), values))
    width = max(900, 105 * len(plot_rows) + 160)
    height = 540
    margin_left = 80
    margin_right = 120
    margin_top = 70
    margin_bottom = 140
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    upper = 0.36
    group_w = plot_w / max(1, len(plot_rows))
    bar_w = min(28, group_w / 3)

    lines = svg_header(width, height, title)
    lines.append(f'<text x="{width/2:.1f}" y="32" text-anchor="middle" class="title">{svg_escape(title)}</text>')
    lines.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" class="axis"/>')
    lines.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" class="axis"/>')
    for tick in range(0, 6):
        value = upper * tick / 5
        y = margin_top + plot_h - (value / upper) * plot_h
        lines.append(f'<line x1="{margin_left-5}" y1="{y:.1f}" x2="{margin_left + plot_w}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{margin_left-10}" y="{y+4:.1f}" text-anchor="end" class="tick">{value:.2f}</text>')
    for index, (label, values) in enumerate(plot_rows):
        center = margin_left + group_w * index + group_w / 2
        for metric_index, value in enumerate(values):
            if value is None:
                continue
            x = center - bar_w - 3 + metric_index * (bar_w + 6)
            bar_h = (value / upper) * plot_h
            y = margin_top + plot_h - bar_h
            color = metrics[metric_index][2]
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="3"/>')
        for line_index, part in enumerate(label.split("\n")):
            lines.append(f'<text x="{center:.1f}" y="{margin_top + plot_h + 28 + 16*line_index:.1f}" text-anchor="middle" class="label">{svg_escape(part)}</text>')
    for idx, (_, label, color) in enumerate(metrics):
        y = margin_top + 10 + idx * 24
        lines.append(f'<rect x="{width - margin_right + 10}" y="{y}" width="14" height="14" fill="{color}"/>')
        lines.append(f'<text x="{width - margin_right + 30}" y="{y+12}" class="label">{svg_escape(label)}</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_svg_scatter(path: Path, title: str, rows: list[dict[str, str]]) -> None:
    points = []
    for row in rows:
        latency = safe_float(row.get("mean_latency_s"))
        bert = safe_float(row.get("bertscore_f1_mean"))
        if latency is not None and bert is not None:
            points.append((short_name(row), latency, bert))
    if not points:
        return
    width = 820
    height = 500
    margin_left = 80
    margin_right = 50
    margin_top = 70
    margin_bottom = 75
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    x_max = max(latency for _, latency, _ in points) * 1.15
    y_min = min(bert for _, _, bert in points) - 0.03
    y_max = max(bert for _, _, bert in points) + 0.03

    lines = svg_header(width, height, title)
    lines.append(f'<text x="{width/2:.1f}" y="32" text-anchor="middle" class="title">{svg_escape(title)}</text>')
    lines.append(f'<text x="{margin_left + plot_w/2:.1f}" y="{height-22}" text-anchor="middle" class="axis-label">Mean latency (seconds/example)</text>')
    lines.append(f'<text x="22" y="{margin_top + plot_h/2:.1f}" transform="rotate(-90 22 {margin_top + plot_h/2:.1f})" text-anchor="middle" class="axis-label">BERTScore F1</text>')
    lines.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" class="axis"/>')
    lines.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" class="axis"/>')
    for tick in range(0, 6):
        x_value = x_max * tick / 5
        x = margin_left + (x_value / x_max) * plot_w
        lines.append(f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + plot_h + 5}" class="grid"/>')
        lines.append(f'<text x="{x:.1f}" y="{margin_top + plot_h + 22}" text-anchor="middle" class="tick">{x_value:.0f}</text>')
        y_value = y_min + (y_max - y_min) * tick / 5
        y = margin_top + plot_h - ((y_value - y_min) / (y_max - y_min)) * plot_h
        lines.append(f'<line x1="{margin_left-5}" y1="{y:.1f}" x2="{margin_left + plot_w}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{margin_left-10}" y="{y+4:.1f}" text-anchor="end" class="tick">{y_value:.2f}</text>')
    colors = ["#4C78A8", "#E45756", "#54A24B"]
    for index, (label, latency, bert) in enumerate(points):
        x = margin_left + (latency / x_max) * plot_w
        y = margin_top + plot_h - ((bert - y_min) / (y_max - y_min)) * plot_h
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{colors[index % len(colors)]}"/>')
        lines.append(f'<text x="{x+10:.1f}" y="{y-8:.1f}" class="label">{svg_escape(label.replace(chr(10), " "))}</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_svg_training_curve(path: Path) -> None:
    history_path = PROJECT_ROOT / "outputs" / "results" / "dataset_enriched" / "vlm_lora" / "training_log_history.csv"
    rows = read_csv(history_path)
    train_points: list[tuple[float, float]] = []
    eval_points: list[tuple[float, float]] = []
    for row in rows:
        step = safe_float(row.get("step"))
        if step is None:
            continue
        loss = safe_float(row.get("loss"))
        eval_loss = safe_float(row.get("eval_loss"))
        if loss is not None:
            train_points.append((step, loss))
        if eval_loss is not None:
            eval_points.append((step, eval_loss))
    if not train_points and not eval_points:
        return

    all_points = train_points + eval_points
    x_max = max(step for step, _ in all_points)
    y_max = max(value for _, value in all_points) * 1.1
    y_min = max(0.0, min(value for _, value in all_points) * 0.85)
    width = 820
    height = 480
    margin_left = 80
    margin_right = 120
    margin_top = 70
    margin_bottom = 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    def point_xy(point: tuple[float, float]) -> tuple[float, float]:
        step, value = point
        x = margin_left + (step / x_max) * plot_w
        y = margin_top + plot_h - ((value - y_min) / (y_max - y_min)) * plot_h
        return x, y

    def polyline(points: list[tuple[float, float]], color: str) -> None:
        if not points:
            return
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in (point_xy(point) for point in points))
        lines.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.6"/>')
        for point in points:
            x, y = point_xy(point)
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>')

    lines = svg_header(width, height, "QLoRA training curve: dataset_enriched")
    lines.append(f'<text x="{width/2:.1f}" y="32" text-anchor="middle" class="title">QLoRA training curve: dataset_enriched</text>')
    lines.append(f'<text x="{margin_left + plot_w/2:.1f}" y="{height-22}" text-anchor="middle" class="axis-label">Step</text>')
    lines.append(f'<text x="22" y="{margin_top + plot_h/2:.1f}" transform="rotate(-90 22 {margin_top + plot_h/2:.1f})" text-anchor="middle" class="axis-label">Loss</text>')
    lines.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" class="axis"/>')
    lines.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" class="axis"/>')
    for tick in range(0, 6):
        x_value = x_max * tick / 5
        x = margin_left + (x_value / x_max) * plot_w
        lines.append(f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + plot_h}" class="grid"/>')
        lines.append(f'<text x="{x:.1f}" y="{margin_top + plot_h + 20}" text-anchor="middle" class="tick">{x_value:.0f}</text>')
        y_value = y_min + (y_max - y_min) * tick / 5
        y = margin_top + plot_h - ((y_value - y_min) / (y_max - y_min)) * plot_h
        lines.append(f'<line x1="{margin_left-5}" y1="{y:.1f}" x2="{margin_left + plot_w}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{margin_left-10}" y="{y+4:.1f}" text-anchor="end" class="tick">{y_value:.2f}</text>')
    polyline(train_points, "#4C78A8")
    polyline(eval_points, "#E45756")
    lines.append(f'<rect x="{width - margin_right + 10}" y="{margin_top}" width="14" height="14" fill="#4C78A8"/>')
    lines.append(f'<text x="{width - margin_right + 30}" y="{margin_top+12}" class="label">train loss</text>')
    lines.append(f'<rect x="{width - margin_right + 10}" y="{margin_top+26}" width="14" height="14" fill="#E45756"/>')
    lines.append(f'<text x="{width - margin_right + 30}" y="{margin_top+38}" class="label">eval loss</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def quantiles(values: list[float]) -> tuple[float, float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    ordered = sorted(values)

    def q(position: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        index = position * (len(ordered) - 1)
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[int(index)]
        return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)

    return ordered[0], q(0.25), q(0.5), q(0.75), ordered[-1]


def write_svg_metric_distributions(path: Path) -> None:
    rows = read_csv(PROJECT_ROOT / "outputs" / "metrics" / "dataset_enriched" / "per_case_vlm_lora_test.csv")
    metrics = [
        ("chrf", "chrF"),
        ("rouge_l", "ROUGE-L"),
        ("token_f1", "Token F1"),
        ("bertscore_f1", "BERTScore"),
    ]
    distributions = []
    for key, label in metrics:
        values = [safe_float(row.get(key)) for row in rows]
        distributions.append((label, [value for value in values if value is not None]))
    width = 820
    height = 500
    margin_left = 80
    margin_right = 40
    margin_top = 70
    margin_bottom = 90
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    upper = 0.85
    group_w = plot_w / len(distributions)
    box_w = 60
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    lines = svg_header(width, height, "Metric distributions: enriched VLM LoRA test")
    lines.append(f'<text x="{width/2:.1f}" y="32" text-anchor="middle" class="title">Metric distributions: enriched VLM LoRA test</text>')
    lines.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" class="axis"/>')
    lines.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" class="axis"/>')
    for tick in range(0, 6):
        value = upper * tick / 5
        y = margin_top + plot_h - (value / upper) * plot_h
        lines.append(f'<line x1="{margin_left-5}" y1="{y:.1f}" x2="{margin_left + plot_w}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{margin_left-10}" y="{y+4:.1f}" text-anchor="end" class="tick">{value:.2f}</text>')
    for index, (label, values) in enumerate(distributions):
        if not values:
            continue
        low, q1, median, q3, high = quantiles(values)
        center = margin_left + group_w * index + group_w / 2
        y_low = margin_top + plot_h - (low / upper) * plot_h
        y_q1 = margin_top + plot_h - (q1 / upper) * plot_h
        y_med = margin_top + plot_h - (median / upper) * plot_h
        y_q3 = margin_top + plot_h - (q3 / upper) * plot_h
        y_high = margin_top + plot_h - (high / upper) * plot_h
        color = colors[index % len(colors)]
        lines.append(f'<line x1="{center:.1f}" y1="{y_high:.1f}" x2="{center:.1f}" y2="{y_low:.1f}" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<rect x="{center-box_w/2:.1f}" y="{y_q3:.1f}" width="{box_w}" height="{max(1, y_q1-y_q3):.1f}" fill="{color}" opacity="0.35" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<line x1="{center-box_w/2:.1f}" y1="{y_med:.1f}" x2="{center+box_w/2:.1f}" y2="{y_med:.1f}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{center:.1f}" y="{margin_top + plot_h + 28}" text-anchor="middle" class="label">{svg_escape(label)}</text>')
        lines.append(f'<text x="{center:.1f}" y="{margin_top + plot_h + 46}" text-anchor="middle" class="tick">med={median:.2f}</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def word_count(text: str) -> int:
    return len(str(text or "").split())


def write_svg_answer_length_alignment(path: Path) -> None:
    rows = read_csv(PROJECT_ROOT / "outputs" / "results" / "dataset_enriched" / "vlm_lora" / "predictions_test.csv")
    pairs = [(word_count(row.get("reference_answer_es", "")), word_count(row.get("predicted_answer_es", ""))) for row in rows]
    if not pairs:
        return
    width = 760
    height = 520
    margin_left = 80
    margin_right = 50
    margin_top = 70
    margin_bottom = 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    upper = max(max(ref, pred) for ref, pred in pairs) * 1.1
    lines = svg_header(width, height, "Answer length alignment: enriched VLM LoRA test")
    lines.append(f'<text x="{width/2:.1f}" y="32" text-anchor="middle" class="title">Answer length alignment: enriched VLM LoRA test</text>')
    lines.append(f'<text x="{margin_left + plot_w/2:.1f}" y="{height-22}" text-anchor="middle" class="axis-label">Reference words</text>')
    lines.append(f'<text x="22" y="{margin_top + plot_h/2:.1f}" transform="rotate(-90 22 {margin_top + plot_h/2:.1f})" text-anchor="middle" class="axis-label">Prediction words</text>')
    lines.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" class="axis"/>')
    lines.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" class="axis"/>')
    lines.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top}" stroke="#999" stroke-dasharray="5,5"/>')
    for tick in range(0, 6):
        value = upper * tick / 5
        x = margin_left + (value / upper) * plot_w
        y = margin_top + plot_h - (value / upper) * plot_h
        lines.append(f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + plot_h}" class="grid"/>')
        lines.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_w}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{x:.1f}" y="{margin_top + plot_h + 20}" text-anchor="middle" class="tick">{value:.0f}</text>')
        lines.append(f'<text x="{margin_left-10}" y="{y+4:.1f}" text-anchor="end" class="tick">{value:.0f}</text>')
    for ref_count, pred_count in pairs:
        x = margin_left + (ref_count / upper) * plot_w
        y = margin_top + plot_h - (pred_count / upper) * plot_h
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#4C78A8" opacity="0.55"/>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_dataset_split_chart(path: Path, rows: list[dict[str, str]]) -> None:
    datasets = ["dataset_longest_answer", "dataset_enriched"]
    splits = ["train", "valid", "test"]
    lookup = {(row["dataset_variant"], row["split"]): int(row["image_row_count"]) for row in rows}
    width = 820
    height = 480
    margin_left = 80
    margin_right = 40
    margin_top = 70
    margin_bottom = 100
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    upper = max(lookup.values()) * 1.15
    group_w = plot_w / len(datasets)
    bar_w = 34
    colors = {"train": "#4C78A8", "valid": "#F58518", "test": "#54A24B"}
    labels = {
        "dataset_longest_answer": "Longest",
        "dataset_enriched": "Enriched",
    }
    lines = svg_header(width, height, "Dataset sizes by split")
    lines.append(f'<text x="{width/2:.1f}" y="32" text-anchor="middle" class="title">Dataset sizes by split</text>')
    lines.append(f'<text x="22" y="{margin_top + plot_h/2:.1f}" transform="rotate(-90 22 {margin_top + plot_h/2:.1f})" text-anchor="middle" class="axis-label">Rows / image rows</text>')
    lines.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" class="axis"/>')
    lines.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" class="axis"/>')
    for tick in range(0, 6):
        value = upper * tick / 5
        y = margin_top + plot_h - (value / upper) * plot_h
        lines.append(f'<line x1="{margin_left-5}" y1="{y:.1f}" x2="{margin_left + plot_w}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{margin_left-10}" y="{y+4:.1f}" text-anchor="end" class="tick">{value:.0f}</text>')
    for dataset_index, dataset in enumerate(datasets):
        center = margin_left + group_w * dataset_index + group_w / 2
        for split_index, split in enumerate(splits):
            value = lookup.get((dataset, split), 0)
            x = center - bar_w * 1.5 - 8 + split_index * (bar_w + 8)
            bar_h = (value / upper) * plot_h
            y = margin_top + plot_h - bar_h
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="{colors[split]}" rx="3"/>')
        lines.append(f'<text x="{center:.1f}" y="{margin_top + plot_h + 30}" text-anchor="middle" class="label">{labels[dataset]}</text>')
    for index, split in enumerate(splits):
        x = width - margin_right - 95
        y = margin_top + index * 24
        lines.append(f'<rect x="{x}" y="{y}" width="14" height="14" fill="{colors[split]}"/>')
        lines.append(f'<text x="{x+22}" y="{y+12}" class="label">{split}</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{svg_escape(title)}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#222}.title{font-size:22px;font-weight:700}.axis-label{font-size:14px;font-weight:600}.tick{font-size:11px;fill:#555}.label{font-size:12px}.value{font-size:12px;font-weight:700}.axis{stroke:#333;stroke-width:1.3}.grid{stroke:#ddd;stroke-width:1}",
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
    ]


def write_markdown_table(path: Path, title: str, rows: list[dict[str, str]]) -> None:
    columns = [
        "dataset_variant",
        "method",
        "split",
        "unit",
        "n",
        "sacrebleu",
        "chrf_corpus",
        "chrf_mean",
        "rouge_l_mean",
        "token_f1_mean",
        "bertscore_f1_mean",
        "mean_latency_s",
    ]
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(svg_escape(row.get(column, "")) for column in columns) + " |")
    lines.append("")
    lines.append("Notes:")
    lines.append("- `chrf_corpus` is on a 0-100 scale when copied from sacreBLEU corpus chrF.")
    lines.append("- `chrf_mean`, `ROUGE-L`, `token-F1`, and `BERTScore F1` are on a 0-1 scale.")
    lines.append("- The main table uses case-level rows; raw image-level enriched VLM rows remain in the long metrics table.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def truncate_text(text: str, limit: int = 280) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "…"


def is_generic_prediction(text: str) -> bool:
    normalized = normalize_text(text)
    generic_markers = [
        "consulte a un dermatólogo",
        "consulta con un dermatólogo",
        "no puedo diagnosticar",
        "no se puede determinar",
        "la imagen no es clara",
    ]
    return any(marker in normalized for marker in generic_markers)


PRELIMINARY_REVIEW_BY_ENCOUNTER: dict[str, dict[str, str]] = {
    "ENC00908": {
        "clinical_correctness": "incorrect",
        "diagnosis_supported": "no",
        "recommendation_safety": "unsupported",
        "hallucination_or_invented_info": "yes",
        "genericness": "specific_wrong",
        "query_contradiction": "yes",
        "image_contradiction": "not_assessed",
        "image_usefulness": "image_critical",
        "text_vs_image_dependency": "mixed",
        "spanish_tone": "good",
        "severity": "high",
        "reviewer_notes": "AI-preliminary: reference supports foliculitis/syphilis/dishidrosis, but prediction shifts to urticaria/acne/psoriasis and adds fungal pathology.",
    },
    "ENC00909": {
        "clinical_correctness": "incorrect",
        "diagnosis_supported": "no",
        "recommendation_safety": "unsupported",
        "hallucination_or_invented_info": "yes",
        "genericness": "specific_wrong",
        "query_contradiction": "yes",
        "image_contradiction": "not_assessed",
        "image_usefulness": "image_critical",
        "text_vs_image_dependency": "image_dominant",
        "spanish_tone": "good",
        "severity": "high",
        "reviewer_notes": "AI-preliminary: reference lists linfangioma/nevus/penfigoide; prediction gives psoriasis and unrelated differentials.",
    },
    "ENC00910": {
        "clinical_correctness": "partial",
        "diagnosis_supported": "partial",
        "recommendation_safety": "mostly_supported",
        "hallucination_or_invented_info": "partial",
        "genericness": "not_generic",
        "query_contradiction": "partial",
        "image_contradiction": "not_assessed",
        "image_usefulness": "mixed",
        "text_vs_image_dependency": "text_dominant",
        "spanish_tone": "good",
        "severity": "medium",
        "reviewer_notes": "AI-preliminary: misses primary tinea framing but preserves fungal confirmation and antifungal direction; adds psoriasis/contact dermatitis.",
    },
    "ENC00935": {
        "clinical_correctness": "incorrect",
        "diagnosis_supported": "no",
        "recommendation_safety": "potentially_unsafe",
        "hallucination_or_invented_info": "yes",
        "genericness": "specific_wrong",
        "query_contradiction": "yes",
        "image_contradiction": "not_assessed",
        "image_usefulness": "image_critical",
        "text_vs_image_dependency": "mixed",
        "spanish_tone": "acceptable",
        "severity": "high",
        "reviewer_notes": "AI-preliminary: reference says liquen plano; prediction invents drug eruption workup and antibiotics/antihistamines.",
    },
    "ENC00950": {
        "clinical_correctness": "partial",
        "diagnosis_supported": "partial",
        "recommendation_safety": "unsupported",
        "hallucination_or_invented_info": "partial",
        "genericness": "not_generic",
        "query_contradiction": "partial",
        "image_contradiction": "not_assessed",
        "image_usefulness": "image_critical",
        "text_vs_image_dependency": "mixed",
        "spanish_tone": "good",
        "severity": "medium",
        "reviewer_notes": "AI-preliminary: captures liquenoid/verruca possibilities but misses hypopigmented nevus/lichen striatus and adds fungal testing/treatment.",
    },
    "ENC01003": {
        "clinical_correctness": "partial",
        "diagnosis_supported": "partial",
        "recommendation_safety": "unsupported",
        "hallucination_or_invented_info": "partial",
        "genericness": "not_generic",
        "query_contradiction": "partial",
        "image_contradiction": "not_assessed",
        "image_usefulness": "mixed",
        "text_vs_image_dependency": "text_dominant",
        "spanish_tone": "good",
        "severity": "medium",
        "reviewer_notes": "AI-preliminary: includes dermatitis de contacto but reframes as drug eruption and adds broad blood/liver testing not in reference.",
    },
    "ENC00938": {
        "clinical_correctness": "partial",
        "diagnosis_supported": "partial",
        "recommendation_safety": "mostly_supported",
        "hallucination_or_invented_info": "partial",
        "genericness": "not_generic",
        "query_contradiction": "partial",
        "image_contradiction": "not_assessed",
        "image_usefulness": "image_critical",
        "text_vs_image_dependency": "mixed",
        "spanish_tone": "good",
        "severity": "medium",
        "reviewer_notes": "AI-preliminary: covers eczema possibility for abdomen but misses necrobiosis lipoidea and herpes-zoster disagreement.",
    },
    "ENC00961": {
        "clinical_correctness": "incorrect",
        "diagnosis_supported": "no",
        "recommendation_safety": "unsupported",
        "hallucination_or_invented_info": "yes",
        "genericness": "not_generic",
        "query_contradiction": "yes",
        "image_contradiction": "not_assessed",
        "image_usefulness": "image_critical",
        "text_vs_image_dependency": "mixed",
        "spanish_tone": "good",
        "severity": "high",
        "reviewer_notes": "AI-preliminary: reference supports psoriasis; prediction accepts eczema numular/Malassezia/fungal route.",
    },
    "ENC00924": {
        "clinical_correctness": "incorrect",
        "diagnosis_supported": "no",
        "recommendation_safety": "potentially_unsafe",
        "hallucination_or_invented_info": "yes",
        "genericness": "specific_wrong",
        "query_contradiction": "yes",
        "image_contradiction": "not_assessed",
        "image_usefulness": "mixed",
        "text_vs_image_dependency": "text_dominant",
        "spanish_tone": "acceptable",
        "severity": "high",
        "reviewer_notes": "AI-preliminary: reference suggests Sweet syndrome; prediction switches to drug/fungal/vitamin workup and antifungals.",
    },
    "ENC00988": {
        "clinical_correctness": "partial",
        "diagnosis_supported": "partial",
        "recommendation_safety": "unsupported",
        "hallucination_or_invented_info": "partial",
        "genericness": "not_generic",
        "query_contradiction": "partial",
        "image_contradiction": "not_assessed",
        "image_usefulness": "image_critical",
        "text_vs_image_dependency": "mixed",
        "spanish_tone": "good",
        "severity": "medium",
        "reviewer_notes": "AI-preliminary: includes verruga plana differential but misses primary siringoma and adds treatment details not in reference.",
    },
    "ENC00986": {
        "clinical_correctness": "correct",
        "diagnosis_supported": "yes",
        "recommendation_safety": "supported",
        "hallucination_or_invented_info": "minor",
        "genericness": "not_generic",
        "query_contradiction": "no",
        "image_contradiction": "not_assessed",
        "image_usefulness": "mixed",
        "text_vs_image_dependency": "mixed",
        "spanish_tone": "good",
        "severity": "low",
        "reviewer_notes": "AI-preliminary: matches psoriasis and biopsy confirmation; dermatitis contact differential is extra but low risk.",
    },
    "ENC00932": {
        "clinical_correctness": "partial",
        "diagnosis_supported": "yes",
        "recommendation_safety": "supported",
        "hallucination_or_invented_info": "minor",
        "genericness": "not_generic",
        "query_contradiction": "no",
        "image_contradiction": "not_assessed",
        "image_usefulness": "image_critical",
        "text_vs_image_dependency": "mixed",
        "spanish_tone": "good",
        "severity": "low",
        "reviewer_notes": "AI-preliminary: aligns with possible onychomycosis and fungal confirmation; omits uncertainty that it may not be onychomycosis.",
    },
    "ENC00976": {
        "clinical_correctness": "correct",
        "diagnosis_supported": "yes",
        "recommendation_safety": "supported",
        "hallucination_or_invented_info": "no",
        "genericness": "not_generic",
        "query_contradiction": "no",
        "image_contradiction": "not_assessed",
        "image_usefulness": "mixed",
        "text_vs_image_dependency": "text_dominant",
        "spanish_tone": "good",
        "severity": "low",
        "reviewer_notes": "AI-preliminary: matches Malassezia folliculitis and fungal confirmation; reasonable differential with acne.",
    },
    "ENC00927": {
        "clinical_correctness": "partial",
        "diagnosis_supported": "partial",
        "recommendation_safety": "mostly_supported",
        "hallucination_or_invented_info": "minor",
        "genericness": "not_generic",
        "query_contradiction": "partial",
        "image_contradiction": "not_assessed",
        "image_usefulness": "image_critical",
        "text_vs_image_dependency": "mixed",
        "spanish_tone": "good",
        "severity": "medium",
        "reviewer_notes": "AI-preliminary: includes eczema differential but promotes psoriasis over reference's eczema chronic framing.",
    },
    "ENC00960": {
        "clinical_correctness": "partial",
        "diagnosis_supported": "yes",
        "recommendation_safety": "mostly_supported",
        "hallucination_or_invented_info": "partial",
        "genericness": "not_generic",
        "query_contradiction": "no",
        "image_contradiction": "not_assessed",
        "image_usefulness": "image_critical",
        "text_vs_image_dependency": "mixed",
        "spanish_tone": "good",
        "severity": "medium",
        "reviewer_notes": "AI-preliminary: matches eczema/neurodermatitis direction but adds psoriasis/contact/keratosis and biopsy not in reference.",
    },
    "ENC00944": {
        "clinical_correctness": "correct",
        "diagnosis_supported": "yes",
        "recommendation_safety": "supported",
        "hallucination_or_invented_info": "minor",
        "genericness": "not_generic",
        "query_contradiction": "no",
        "image_contradiction": "not_assessed",
        "image_usefulness": "mixed",
        "text_vs_image_dependency": "text_dominant",
        "spanish_tone": "good",
        "severity": "low",
        "reviewer_notes": "AI-preliminary: matches urticaria papular, allergy framing and allergen testing; adds common differentials.",
    },
    "ENC00967": {
        "clinical_correctness": "partial",
        "diagnosis_supported": "yes",
        "recommendation_safety": "supported",
        "hallucination_or_invented_info": "minor",
        "genericness": "not_generic",
        "query_contradiction": "no",
        "image_contradiction": "not_assessed",
        "image_usefulness": "mixed",
        "text_vs_image_dependency": "text_dominant",
        "spanish_tone": "good",
        "severity": "low",
        "reviewer_notes": "AI-preliminary: preserves vasculitis/globulinemia-cold concept but broadens into cold eruption and lab work.",
    },
    "ENC00989": {
        "clinical_correctness": "incorrect",
        "diagnosis_supported": "no",
        "recommendation_safety": "unsupported",
        "hallucination_or_invented_info": "yes",
        "genericness": "not_generic",
        "query_contradiction": "yes",
        "image_contradiction": "not_assessed",
        "image_usefulness": "image_critical",
        "text_vs_image_dependency": "mixed",
        "spanish_tone": "good",
        "severity": "high",
        "reviewer_notes": "AI-preliminary: reference supports angioma/linfangioma; prediction gives granuloma anular/folliculitis/verruca/fungal.",
    },
    "ENC01004": {
        "clinical_correctness": "incorrect",
        "diagnosis_supported": "no",
        "recommendation_safety": "unsupported",
        "hallucination_or_invented_info": "yes",
        "genericness": "not_generic",
        "query_contradiction": "yes",
        "image_contradiction": "not_assessed",
        "image_usefulness": "mixed",
        "text_vs_image_dependency": "mixed",
        "spanish_tone": "good",
        "severity": "high",
        "reviewer_notes": "AI-preliminary: reference supports actinic dermatitis/keratosis/liquenification; prediction shifts to urticaria papular/allergy and antihistamines.",
    },
    "ENC00987": {
        "clinical_correctness": "incorrect",
        "diagnosis_supported": "no",
        "recommendation_safety": "unsupported",
        "hallucination_or_invented_info": "yes",
        "genericness": "not_generic",
        "query_contradiction": "yes",
        "image_contradiction": "not_assessed",
        "image_usefulness": "image_critical",
        "text_vs_image_dependency": "mixed",
        "spanish_tone": "good",
        "severity": "high",
        "reviewer_notes": "AI-preliminary: reference is pigmented hairy epidermal nevus; prediction gives flat warts/Malassezia/fungal testing.",
    },
}


def write_clinical_review_sheet(tables_dir: Path) -> None:
    metrics = read_csv(PROJECT_ROOT / "outputs" / "metrics" / "dataset_enriched" / "per_case_vlm_lora_test.csv")
    predictions = read_csv(PROJECT_ROOT / "outputs" / "results" / "dataset_enriched" / "vlm_lora" / "predictions_test.csv")
    prediction_lookup = {
        (row.get("encounter_id", ""), row.get("image_id", "")): row
        for row in predictions
    }
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in metrics:
        encounter_id = row.get("encounter_id", "")
        if encounter_id:
            grouped[encounter_id].append(row)

    case_items: list[dict[str, Any]] = []
    for encounter_id, rows in grouped.items():
        metric_values: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            for metric in ["chrf", "rouge_l", "token_f1", "bertscore_f1"]:
                value = safe_float(row.get(metric))
                if value is not None:
                    metric_values[metric].append(value)
        avg_bert = sum(metric_values["bertscore_f1"]) / len(metric_values["bertscore_f1"]) if metric_values["bertscore_f1"] else 0.0

        def distance_to_average(row: dict[str, str]) -> float:
            value = safe_float(row.get("bertscore_f1"))
            return abs(value - avg_bert) if value is not None else float("inf")

        representative = sorted(rows, key=lambda row: (distance_to_average(row), row.get("image_id", "")))[0]
        prediction = prediction_lookup.get(
            (representative.get("encounter_id", ""), representative.get("image_id", "")),
            {},
        )
        case_items.append(
            {
                "encounter_id": encounter_id,
                "representative": representative,
                "prediction": prediction,
                "avg_chrf": sum(metric_values["chrf"]) / len(metric_values["chrf"]) if metric_values["chrf"] else None,
                "avg_rouge_l": sum(metric_values["rouge_l"]) / len(metric_values["rouge_l"]) if metric_values["rouge_l"] else None,
                "avg_token_f1": sum(metric_values["token_f1"]) / len(metric_values["token_f1"]) if metric_values["token_f1"] else None,
                "avg_bertscore_f1": avg_bert,
            }
        )

    item_by_case = {item["encounter_id"]: item for item in case_items}
    selected: list[tuple[str, dict[str, Any]]] = []
    selected_ids: set[str] = set()

    def add_item(bucket: str, item: dict[str, Any] | None) -> None:
        if not item:
            return
        encounter_id = item["encounter_id"]
        if encounter_id in selected_ids:
            return
        selected.append((bucket, item))
        selected_ids.add(encounter_id)

    for encounter_id in ["ENC00853", "ENC00854", "ENC00908", "ENC00909", "ENC00910"]:
        add_item("survey_named_case", item_by_case.get(encounter_id))

    sorted_by_bert = sorted(case_items, key=lambda item: item["avg_bertscore_f1"])
    for item in sorted_by_bert[:7]:
        add_item("low_metric_case", item)
    for item in reversed(sorted_by_bert[-7:]):
        add_item("high_metric_case", item)

    median_bert = sorted_by_bert[len(sorted_by_bert) // 2]["avg_bertscore_f1"] if sorted_by_bert else 0.0
    for item in sorted(case_items, key=lambda item: abs(item["avg_bertscore_f1"] - median_bert)):
        add_item("middle_metric_case", item)
        if len(selected) >= 20:
            break

    rows: list[dict[str, str]] = []
    for bucket, item in selected[:20]:
        metric_row_item = item["representative"]
        prediction = item["prediction"]
        predicted_answer = prediction.get("predicted_answer_es", "")
        review = PRELIMINARY_REVIEW_BY_ENCOUNTER.get(item["encounter_id"], {})
        rows.append(
            {
                "case_bucket": bucket,
                "dataset_variant": "dataset_enriched",
                "model": "Qwen/Qwen2.5-VL-3B-Instruct + LoRA",
                "unit": "case_review_representative_image",
                "split": "test",
                "encounter_id": item["encounter_id"],
                "representative_image_id": metric_row_item.get("image_id", ""),
                "image_path": prediction.get("image_path", ""),
                "avg_chrf": fmt(item.get("avg_chrf")),
                "avg_rouge_l": fmt(item.get("avg_rouge_l")),
                "avg_token_f1": fmt(item.get("avg_token_f1")),
                "avg_bertscore_f1": fmt(item.get("avg_bertscore_f1")),
                "question_es": truncate_text(prediction.get("question_es", ""), 360),
                "reference_answer_es": truncate_text(prediction.get("reference_answer_es", ""), 420),
                "predicted_answer_es": truncate_text(predicted_answer, 420),
                "auto_empty_prediction": str(not bool(predicted_answer.strip())).lower(),
                "auto_short_prediction": str(word_count(predicted_answer) < 8).lower(),
                "auto_generic_prediction": str(is_generic_prediction(predicted_answer)).lower(),
                "reviewer_type": "ai_preliminary",
                "review_status": "needs_clinician_confirmation",
                "clinical_correctness": review.get("clinical_correctness", ""),
                "diagnosis_supported": review.get("diagnosis_supported", ""),
                "recommendation_safety": review.get("recommendation_safety", ""),
                "hallucination_or_invented_info": review.get("hallucination_or_invented_info", ""),
                "genericness": review.get("genericness", ""),
                "query_contradiction": review.get("query_contradiction", ""),
                "image_contradiction": review.get("image_contradiction", ""),
                "image_usefulness": review.get("image_usefulness", ""),
                "text_vs_image_dependency": review.get("text_vs_image_dependency", ""),
                "spanish_tone": review.get("spanish_tone", ""),
                "severity": review.get("severity", ""),
                "reviewer_notes": review.get("reviewer_notes", ""),
            }
        )

    columns = [
        "case_bucket",
        "dataset_variant",
        "model",
        "unit",
        "split",
        "encounter_id",
        "representative_image_id",
        "image_path",
        "avg_chrf",
        "avg_rouge_l",
        "avg_token_f1",
        "avg_bertscore_f1",
        "question_es",
        "reference_answer_es",
        "predicted_answer_es",
        "auto_empty_prediction",
        "auto_short_prediction",
        "auto_generic_prediction",
        "reviewer_type",
        "review_status",
        "clinical_correctness",
        "diagnosis_supported",
        "recommendation_safety",
        "hallucination_or_invented_info",
        "genericness",
        "query_contradiction",
        "image_contradiction",
        "image_usefulness",
        "text_vs_image_dependency",
        "spanish_tone",
        "severity",
        "reviewer_notes",
    ]
    write_csv(tables_dir / "paper_clinical_review_20.csv", rows, columns)

    summary_rows: list[dict[str, str]] = []
    for field in [
        "clinical_correctness",
        "diagnosis_supported",
        "recommendation_safety",
        "hallucination_or_invented_info",
        "severity",
        "spanish_tone",
    ]:
        counts = Counter(row.get(field, "") for row in rows)
        for label, count in sorted(counts.items()):
            if label:
                summary_rows.append(
                    {
                        "field": field,
                        "label": label,
                        "count": str(count),
                        "rate": fmt(count / len(rows) if rows else None),
                    }
                )
    write_csv(tables_dir / "paper_clinical_review_summary.csv", summary_rows, ["field", "label", "count", "rate"])

    lines = [
        "# Clinical review sheet: 20 enriched VLM LoRA cases",
        "",
        "This sheet is generated automatically. The labels are AI-preliminary and must be confirmed by a clinician before final medical-safety claims.",
        "",
        "Sampling strategy: survey-named cases when available, low metric cases, high metric cases, and middle metric cases. Rows are de-duplicated by `encounter_id`.",
        "",
        "| bucket | encounter | correctness | hallucination | severity | avg BERTScore | reference | prediction |",
        "| --- | --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    svg_escape(row["case_bucket"]),
                    svg_escape(row["encounter_id"]),
                    svg_escape(row["clinical_correctness"]),
                    svg_escape(row["hallucination_or_invented_info"]),
                    svg_escape(row["severity"]),
                    svg_escape(row["avg_bertscore_f1"]),
                    svg_escape(row["reference_answer_es"]),
                    svg_escape(row["predicted_answer_es"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Reviewer columns in CSV: `reviewer_type`, `review_status`, `clinical_correctness`, `diagnosis_supported`, `recommendation_safety`, `hallucination_or_invented_info`, `genericness`, `query_contradiction`, `image_contradiction`, `image_usefulness`, `text_vs_image_dependency`, `spanish_tone`, `severity`, `reviewer_notes`.",
        ]
    )
    (tables_dir / "paper_clinical_review_20.md").write_text("\n".join(lines), encoding="utf-8")

    summary_lines = [
        "# Preliminary clinical review summary",
        "",
        "AI-preliminary labels over the 20-case enriched VLM LoRA review sheet. These counts are for error analysis only and require clinician confirmation.",
        "",
        "| field | label | count | rate |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in summary_rows:
        summary_lines.append(
            f"| {svg_escape(row['field'])} | {svg_escape(row['label'])} | {row['count']} | {row['rate']} |"
        )
    (tables_dir / "paper_clinical_review_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")


def write_qualitative_cases(tables_dir: Path) -> None:
    metrics = read_csv(PROJECT_ROOT / "outputs" / "metrics" / "dataset_enriched" / "per_case_vlm_lora_test.csv")
    predictions = read_csv(PROJECT_ROOT / "outputs" / "results" / "dataset_enriched" / "vlm_lora" / "predictions_test.csv")
    prediction_lookup = {
        (row.get("encounter_id", ""), row.get("image_id", "")): row
        for row in predictions
    }
    scored = [
        row for row in metrics
        if safe_float(row.get("bertscore_f1")) is not None
    ]
    scored.sort(key=lambda row: safe_float(row.get("bertscore_f1")) or 0.0)
    selected = [("low", row) for row in scored[:5]] + [("high", row) for row in scored[-5:]]
    rows: list[dict[str, str]] = []
    for bucket, metric_row_item in selected:
        key = (metric_row_item.get("encounter_id", ""), metric_row_item.get("image_id", ""))
        prediction = prediction_lookup.get(key, {})
        rows.append(
            {
                "bucket": bucket,
                "split": "test",
                "encounter_id": key[0],
                "image_id": key[1],
                "chrf": fmt(safe_float(metric_row_item.get("chrf"))),
                "rouge_l": fmt(safe_float(metric_row_item.get("rouge_l"))),
                "token_f1": fmt(safe_float(metric_row_item.get("token_f1"))),
                "bertscore_f1": fmt(safe_float(metric_row_item.get("bertscore_f1"))),
                "question_es": truncate_text(prediction.get("question_es", "")),
                "reference_answer_es": truncate_text(prediction.get("reference_answer_es", "")),
                "predicted_answer_es": truncate_text(prediction.get("predicted_answer_es", "")),
                "manual_label": "",
                "manual_notes": "",
            }
        )
    columns = [
        "bucket",
        "split",
        "encounter_id",
        "image_id",
        "chrf",
        "rouge_l",
        "token_f1",
        "bertscore_f1",
        "question_es",
        "reference_answer_es",
        "predicted_answer_es",
        "manual_label",
        "manual_notes",
    ]
    write_csv(tables_dir / "paper_qualitative_review_candidates.csv", rows, columns)
    lines = [
        "# Qualitative review candidates",
        "",
        "Generated from `dataset_enriched/vlm_lora` test predictions.",
        "The table includes the 5 lowest and 5 highest BERTScore F1 rows as a starting point for manual clinical review.",
        "",
        "| bucket | encounter | image | BERTScore | reference | prediction |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    svg_escape(row["bucket"]),
                    svg_escape(row["encounter_id"]),
                    svg_escape(row["image_id"]),
                    svg_escape(row["bertscore_f1"]),
                    svg_escape(row["reference_answer_es"]),
                    svg_escape(row["predicted_answer_es"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("Suggested manual labels: `correct`, `partial`, `incorrect`, `unsupported_recommendation`, `hallucination`, `too_generic`.")
    (tables_dir / "paper_qualitative_review_candidates.md").write_text("\n".join(lines), encoding="utf-8")


def write_missing_metrics_report(path: Path, rows: list[dict[str, str]]) -> None:
    missing: dict[str, list[str]] = defaultdict(list)
    required = ["sacrebleu", "chrf_corpus", "chrf_mean", "rouge_l_mean", "token_f1_mean", "bertscore_f1_mean", "mean_latency_s"]
    for row in rows:
        label = f"{row['dataset_variant']} / {row['method']} / {row['split']}"
        for key in required:
            if not row.get(key):
                missing[label].append(key)
    lines = [
        "# Missing metrics and paper caveats",
        "",
        "This file is generated by `python -m src.build_paper_results`.",
        "",
        "## Missing fields",
        "",
    ]
    for label, fields in sorted(missing.items()):
        lines.append(f"- `{label}`: {', '.join(fields)}")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- The main table uses `dataset_enriched/vlm_lora_case_avg` for fair case-level comparison (`n=100` on test); raw image-level enriched VLM metrics remain in `paper_all_metrics_long.csv`.",
            "- For `dataset_longest_answer`, the main table uses leakage-free TF-IDF retrieval against train only when `src.evaluate_retrieval_heldout` has been run; legacy all-split retrieval rows remain in `paper_all_metrics_long.csv` for appendix context.",
            "- Enriched case-level corpus scores concatenate de-duplicated predictions from all images in a case; this avoids oracle image selection but is not identical to averaging per-image metrics.",
            "- BERTScore is not recomputed by this script; rows generated without BERTScore keep that field empty unless a prior metric artifact exists.",
            "- Human clinical review labels still need a reviewer before final medical-safety claims; `paper_clinical_review_20.csv` is the structured review sheet.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_figures_readme(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Paper figures",
                "",
                "Generated by `python -m src.build_paper_results`.",
                "",
                "- `dataset_split_counts.svg`: dataset size by split.",
                "- `main_test_chrf.svg`: main comparison by chrF.",
                "- `main_test_bertscore.svg`: main comparison by BERTScore F1.",
                "- `main_test_rouge_tokenf1.svg`: ROUGE-L and token-F1 grouped bars.",
                "- `vlm_latency_vs_bertscore.svg`: VLM latency/quality trade-off.",
                "- `enriched_vlm_training_curve.svg`: training/eval loss curve.",
                "- `enriched_vlm_metric_distributions.svg`: per-image metric distributions on test.",
                "- `enriched_vlm_answer_length_alignment.svg`: reference vs prediction word counts.",
                "",
                "Figures are SVG so they can be versioned without conflicting with the rule that raw clinical images are never committed.",
            ]
        ),
        encoding="utf-8",
    )


def write_paper_readme(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Paper-ready outputs",
                "",
                "Generated by:",
                "",
                "```bash",
                "python -m src.evaluate_retrieval_heldout --dataset all",
                "python -m src.build_paper_results",
                "```",
                "",
                "## Contents",
                "",
                "- `tables/paper_main_test_comparison.csv`: compact test comparison for the main paper table.",
                "- `tables/paper_all_metrics_long.csv`: longer metric table with valid/test and auxiliary rows.",
                "- `tables/paper_results_summary.md`: Markdown table for quick review.",
                "- `tables/paper_missing_metrics_report.md`: remaining metric gaps and caveats.",
                "- `tables/paper_qualitative_review_candidates.*`: initial cases for manual clinical review.",
                "- `tables/paper_clinical_review_20.*`: balanced 20-case reviewer sheet for the enriched VLM LoRA model.",
                "- `tables/paper_clinical_review_summary.*`: aggregate counts from the AI-preliminary review sheet.",
                "- `figures/*.svg`: versioned, paper-ready SVG figures.",
                "",
                "## Caveat",
                "",
                "The main table is case-level whenever possible. Human clinical review labels still need to be filled before making medical-safety claims.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run() -> None:
    OUTPUT_PAPER_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, str]] = []
    all_rows.extend(build_enriched_retrieval_rows())
    all_rows.extend(build_enriched_vlm_rows())
    all_rows.extend(build_enriched_vlm_case_average_rows())
    all_rows.extend(build_longest_retrieval_rows())
    all_rows.extend(build_retrieval_heldout_rows())
    all_rows.extend(build_longest_vlm_rows())

    main_rows = main_test_rows(all_rows)
    dataset_counts = build_dataset_split_counts()

    write_csv(OUTPUT_TABLES_DIR / "paper_all_metrics_long.csv", all_rows, METRIC_COLUMNS)
    write_csv(OUTPUT_TABLES_DIR / "paper_main_test_comparison.csv", main_rows, METRIC_COLUMNS)
    write_csv(
        OUTPUT_TABLES_DIR / "paper_dataset_split_counts.csv",
        dataset_counts,
        ["dataset_variant", "split", "case_count", "image_row_count"],
    )
    write_markdown_table(OUTPUT_TABLES_DIR / "paper_results_summary.md", "Paper-ready result summary", main_rows)
    write_missing_metrics_report(OUTPUT_TABLES_DIR / "paper_missing_metrics_report.md", main_rows)
    write_qualitative_cases(OUTPUT_TABLES_DIR)
    write_clinical_review_sheet(OUTPUT_TABLES_DIR)

    write_dataset_split_chart(OUTPUT_FIGURES_DIR / "dataset_split_counts.svg", dataset_counts)
    write_svg_bar_chart(OUTPUT_FIGURES_DIR / "main_test_chrf.svg", "Main test comparison: chrF", main_rows, "chrf", "chrF (0-1 normalized)")
    write_svg_bar_chart(OUTPUT_FIGURES_DIR / "main_test_bertscore.svg", "Main test comparison: BERTScore F1", main_rows, "bertscore_f1_mean", "BERTScore F1", max_value=0.82)
    write_svg_grouped_bar_chart(OUTPUT_FIGURES_DIR / "main_test_rouge_tokenf1.svg", "Main test comparison: ROUGE-L and token-F1", main_rows)
    vlm_rows = [row for row in main_rows if row.get("method", "").startswith("vlm_")]
    write_svg_scatter(OUTPUT_FIGURES_DIR / "vlm_latency_vs_bertscore.svg", "VLM latency vs semantic quality", vlm_rows)
    write_svg_training_curve(OUTPUT_FIGURES_DIR / "enriched_vlm_training_curve.svg")
    write_svg_metric_distributions(OUTPUT_FIGURES_DIR / "enriched_vlm_metric_distributions.svg")
    write_svg_answer_length_alignment(OUTPUT_FIGURES_DIR / "enriched_vlm_answer_length_alignment.svg")
    write_figures_readme(OUTPUT_FIGURES_DIR / "README.md")
    write_paper_readme(OUTPUT_PAPER_DIR / "README.md")

    print(f"Wrote {len(all_rows)} metric rows to {OUTPUT_TABLES_DIR}")
    print(f"Wrote {len(main_rows)} main comparison rows")
    print(f"Wrote figures to {OUTPUT_FIGURES_DIR}")


if __name__ == "__main__":
    run()
