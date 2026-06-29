"""
Evaluacion de predicciones VLM/retrieval generativas.

Toma uno o mas CSV de predicciones con columnas:
  - predicted_answer_es
  - reference_answer_es

Si el CSV trae dataset_variant, escribe metricas en:
  outputs/metrics/<dataset_variant>/

Esto permite comparar directamente `dataset_longest_answer` y `dataset_enriched`
con las mismas metricas.

Uso:
    python -m src.evaluate_predictions \
        outputs/results/dataset_longest_answer/vlm_lora/predictions_valid.csv

    python -m src.evaluate_predictions \
        outputs/results/dataset_enriched/vlm_lora/predictions_valid.csv \
        outputs/results/dataset_enriched/vlm_lora/predictions_test.csv

    python -m src.evaluate_predictions <csv> --no-bertscore
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.retrieval_utils import PROJECT_ROOT

METRICS_BASE = PROJECT_ROOT / "outputs" / "metrics"
DEFAULT_DATASET_VARIANT = "dataset_longest_answer"
BERTSCORE_MODEL = "bert-base-multilingual-cased"
E5_MODEL_ID = "intfloat/multilingual-e5-base"


def normalize_dataset_variant(value: Any, csv_path: Path) -> str:
    text = str(value or "").strip()
    if not text:
        path_text = str(csv_path).replace("\\", "/")
        if "dataset_enriched" in path_text:
            return "dataset_enriched"
        if "dataset_longest_answer" in path_text:
            return "dataset_longest_answer"
        return DEFAULT_DATASET_VARIANT
    aliases = {
        "longest_answer": "dataset_longest_answer",
        "dataset_longest_answer": "dataset_longest_answer",
        "enriched": "dataset_enriched",
        "llm_synthesized_answer": "dataset_enriched",
        "dataset_enriched": "dataset_enriched",
    }
    return aliases.get(text, text)


def normalize(text: str) -> str:
    return " ".join(str(text).lower().split())


def token_f1(pred: str, ref: str) -> float:
    from collections import Counter

    pred_tokens = normalize(pred).split()
    ref_tokens = normalize(ref).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if not common:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_rouge_l(predictions: list[str], references: list[str]) -> list[float]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return [
        scorer.score(reference, prediction)["rougeL"].fmeasure
        for prediction, reference in zip(predictions, references)
    ]


def compute_chrf(predictions: list[str], references: list[str]) -> list[float]:
    import sacrebleu

    return [
        sacrebleu.corpus_chrf([prediction], [[reference]]).score / 100.0
        for prediction, reference in zip(predictions, references)
    ]


def compute_bertscore(predictions: list[str], references: list[str]) -> list[float]:
    from bert_score import score as bert_score

    print(f"  Calculando BERTScore ({BERTSCORE_MODEL})...")
    _, _, f1 = bert_score(
        predictions, references, model_type=BERTSCORE_MODEL, lang="es", verbose=False
    )
    return f1.tolist()


def compute_cosine_e5(predictions: list[str], references: list[str]) -> list[float]:
    from sentence_transformers import SentenceTransformer

    print(f"  Calculando cosine similarity ({E5_MODEL_ID})...")
    model = SentenceTransformer(E5_MODEL_ID)
    emb_pred = model.encode([f"query: {prediction}" for prediction in predictions], normalize_embeddings=True)
    emb_ref = model.encode([f"query: {reference}" for reference in references], normalize_embeddings=True)
    return [float(np.dot(pred_vec, ref_vec)) for pred_vec, ref_vec in zip(emb_pred, emb_ref)]


def corpus_scores(predictions: list[str], references: list[str]) -> dict[str, float]:
    import sacrebleu

    bleu = sacrebleu.corpus_bleu(predictions, [references]).score
    chrf = sacrebleu.corpus_chrf(predictions, [references]).score
    return {"sacrebleu_corpus": bleu, "chrf_corpus": chrf}


def evaluate_csv(csv_path: Path, use_bertscore: bool, use_cosine: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(csv_path).fillna({"predicted_answer_es": "", "reference_answer_es": ""})
    required = {"predicted_answer_es", "reference_answer_es"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas en {csv_path}: {missing}")

    method = str(df["method"].iloc[0]) if "method" in df else csv_path.stem
    model_name = str(df["model_name"].iloc[0]) if "model_name" in df else "unknown"
    split = str(df["split"].iloc[0]) if "split" in df else "unknown"
    raw_variant = str(df["dataset_variant"].iloc[0]) if "dataset_variant" in df else ""
    dataset_variant = normalize_dataset_variant(raw_variant, csv_path)

    predictions = df["predicted_answer_es"].astype(str).tolist()
    references = df["reference_answer_es"].astype(str).tolist()
    empty_predictions = sum(1 for prediction in predictions if not prediction.strip())

    print(f"\n[{dataset_variant} | {method}] {len(predictions)} casos ({empty_predictions} predicciones vacias)")

    chrf = compute_chrf(predictions, references)
    rouge_l = compute_rouge_l(predictions, references)
    tok_f1 = [token_f1(prediction, reference) for prediction, reference in zip(predictions, references)]

    per_case = pd.DataFrame(
        {
            "dataset_variant": dataset_variant,
            "method": method,
            "model_name": model_name,
            "split": split,
            "encounter_id": df.get("encounter_id", pd.Series(range(len(df)))),
            "image_id": df.get("image_id", pd.Series([""] * len(df))),
            "chrf": chrf,
            "rouge_l": rouge_l,
            "token_f1": tok_f1,
        }
    )

    summary: dict[str, Any] = {
        "dataset_variant": dataset_variant,
        "method": method,
        "model_name": model_name,
        "split": split,
        "n": len(predictions),
        "empty_predictions": empty_predictions,
        "chrf_mean": float(np.mean(chrf)),
        "rouge_l_mean": float(np.mean(rouge_l)),
        "token_f1_mean": float(np.mean(tok_f1)),
    }
    summary.update(corpus_scores(predictions, references))

    if use_bertscore:
        bertscore = compute_bertscore(predictions, references)
        per_case["bertscore_f1"] = bertscore
        summary["bertscore_f1_mean"] = float(np.mean(bertscore))

    if use_cosine:
        cosine = compute_cosine_e5(predictions, references)
        per_case["cosine_e5"] = cosine
        summary["cosine_e5_mean"] = float(np.mean(cosine))

    return per_case, summary


def write_summary(metrics_root: Path, summaries: list[dict[str, Any]]) -> None:
    metrics_root.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summaries)
    split_seen = set(summary_df["split"].astype(str).tolist())

    if len(split_seen) == 1:
        split = next(iter(split_seen))
        out_path = metrics_root / f"metrics_{split}.csv"
        if out_path.exists():
            previous = pd.read_csv(out_path)
            previous = previous[~previous["method"].isin(summary_df["method"])]
            summary_df = pd.concat([previous, summary_df], ignore_index=True)
        summary_df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"\nResumen -> {out_path}")
    else:
        out_path = metrics_root / "metrics_mixed.csv"
        if out_path.exists():
            previous = pd.read_csv(out_path)
            key_columns = ["method", "model_name", "split"]
            for column in key_columns:
                if column not in previous.columns:
                    previous[column] = ""
            replace_keys = set(zip(*(summary_df[column].astype(str) for column in key_columns)))
            previous_keys = list(zip(*(previous[column].astype(str) for column in key_columns)))
            previous = previous[[key not in replace_keys for key in previous_keys]]
            summary_df = pd.concat([previous, summary_df], ignore_index=True)
        summary_df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"\nResumen (splits mixtos) -> {out_path}")

    print("\n" + "=" * 70)
    print(metrics_root)
    print("=" * 70)
    print(summary_df.to_string(index=False, float_format="{:.4f}".format))


def run(args: argparse.Namespace) -> None:
    summaries_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for prediction_path in args.predictions:
        csv_path = Path(prediction_path)
        per_case, summary = evaluate_csv(
            csv_path, use_bertscore=not args.no_bertscore, use_cosine=args.cosine
        )
        dataset_variant = summary["dataset_variant"]
        metrics_root = METRICS_BASE / dataset_variant
        metrics_root.mkdir(parents=True, exist_ok=True)

        per_case_path = metrics_root / f"per_case_{summary['method']}_{summary['split']}.csv"
        per_case.to_csv(per_case_path, index=False, encoding="utf-8")
        print(f"  Por caso -> {per_case_path}")

        summaries_by_variant[dataset_variant].append(summary)

    for dataset_variant, summaries in summaries_by_variant.items():
        write_summary(METRICS_BASE / dataset_variant, summaries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalua predicciones generativas VLM")
    parser.add_argument("predictions", nargs="+", help="Uno o mas CSV de predicciones")
    parser.add_argument("--no-bertscore", action="store_true", help="Saltear BERTScore")
    parser.add_argument("--cosine", action="store_true", help="Calcular cosine semantica con E5")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
