"""
Evaluación de baselines de retrieval contra las respuestas de referencia.

Para cada archivo de resultados JSON en un directorio de resultados, calcula:
  - ROUGE-L (F1)
  - Token-level F1
  - chrF (sacrebleu)
  - BERTScore F1 (multilingual)
  - Cosine similarity media de los scores de retrieval

La predicción es la respuesta recuperada; la referencia es la respuesta del
caso original en el dataset.

Salida:
  outputs/metrics/<dataset_variant>/metrics_summary.csv   <- una fila por modelo
  outputs/metrics/<dataset_variant>/metrics_per_case.csv  <- una fila por caso/modelo

Uso:
    # Evalúa todos los modelos del dataset longest_answer
    python -m src.evaluate_retrieval --dataset longest_answer

    # Evalúa todos los modelos del dataset short_answer
    python -m src.evaluate_retrieval --dataset short_answer

    # Evalúa un directorio concreto
    python -m src.evaluate_retrieval --results-dir outputs/results/dataset_longest_answer/retrieval_textual_sbert
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sacrebleu
from bert_score import score as bert_score
from rouge_score import rouge_scorer

from src.retrieval_utils import PROJECT_ROOT, load_dataset

# ── paths ──────────────────────────────────────────────────────────────────────

DATASETS = {
    "longest_answer": PROJECT_ROOT / "outputs" / "datasets" / "dataset_longest_answer.json",
    "short_answer": PROJECT_ROOT / "outputs" / "datasets" / "dataset_short_answer.json",
}
RESULTS_ROOT = PROJECT_ROOT / "outputs" / "results"
METRICS_ROOT = PROJECT_ROOT / "outputs" / "metrics"

# Campo de respuesta recuperada según el tipo de resultado
RETRIEVED_ANSWER_KEYS = ["retrieved_answer_es", "retrieved_short_answer_es"]

# Modelo BERTScore multilingüe
BERTSCORE_MODEL = "bert-base-multilingual-cased"


# ── helpers de texto ───────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def token_f1(pred: str, ref: str) -> dict[str, float]:
    pred_tokens = normalize(pred).split()
    ref_tokens = normalize(ref).split()
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    p = len(common) / len(pred_tokens)
    r = len(common) / len(ref_tokens)
    f1 = 2 * p * r / (p + r)
    return {"precision": p, "recall": r, "f1": f1}


# ── carga ──────────────────────────────────────────────────────────────────────

def load_results(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_reference_map(records: list[dict[str, Any]]) -> dict[str, str]:
    return {r["encounter_id"]: r.get("answer_es", "") for r in records}


def build_split_map(records: list[dict[str, Any]]) -> dict[str, str]:
    return {r["encounter_id"]: r.get("_split", "unknown") for r in records}


def get_retrieved_answer(result: dict[str, Any]) -> str:
    for key in RETRIEVED_ANSWER_KEYS:
        if key in result:
            return result[key]
    return ""


# ── métricas por caso ──────────────────────────────────────────────────────────

def compute_rouge_l(predictions: list[str], references: list[str]) -> list[float]:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for pred, ref in zip(predictions, references)
    ]


def compute_chrf(predictions: list[str], references: list[str]) -> list[float]:
    return [
        sacrebleu.corpus_chrf([pred], [[ref]]).score / 100.0
        for pred, ref in zip(predictions, references)
    ]


def compute_token_f1(predictions: list[str], references: list[str]) -> list[float]:
    return [token_f1(p, r)["f1"] for p, r in zip(predictions, references)]


def compute_bertscore(
    predictions: list[str], references: list[str]
) -> list[float]:
    print(f"  Calculando BERTScore ({BERTSCORE_MODEL})...")
    _, _, f1 = bert_score(
        predictions,
        references,
        model_type=BERTSCORE_MODEL,
        lang="es",
        verbose=False,
    )
    return f1.tolist()


# ── evaluación de un archivo ───────────────────────────────────────────────────

def evaluate_file(
    results_path: Path,
    reference_map: dict[str, str],
    split_map: dict[str, str],
    model_name: str,
    dataset_variant: str,
) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    """
    Retorna (per_case_df, list_of_summary_dicts).
    Hay una fila de summary por split (train/valid/test) más una fila "all".
    """
    results = load_results(results_path)

    encounter_ids: list[str] = []
    predictions: list[str] = []
    references: list[str] = []
    sim_scores: list[float] = []
    splits: list[str] = []

    for r in results:
        eid = r["encounter_id"]
        ref = reference_map.get(eid, "")
        pred = get_retrieved_answer(r)
        if not pred or not ref:
            continue
        encounter_ids.append(eid)
        predictions.append(pred)
        references.append(ref)
        sim_scores.append(r.get("similarity_score", float("nan")))
        splits.append(split_map.get(eid, "unknown"))

    print(f"\n[{model_name}] {len(predictions)} casos evaluados")

    rouge_l = compute_rouge_l(predictions, references)
    chrf = compute_chrf(predictions, references)
    tok_f1 = compute_token_f1(predictions, references)
    bscore = compute_bertscore(predictions, references)

    per_case = pd.DataFrame(
        {
            "dataset_variant": dataset_variant,
            "model": model_name,
            "split": splits,
            "encounter_id": encounter_ids,
            "similarity_score": sim_scores,
            "rouge_l": rouge_l,
            "chrf": chrf,
            "token_f1": tok_f1,
            "bertscore_f1": bscore,
        }
    )

    def _summary_for(mask: pd.Series, split_label: str) -> dict:
        sub = per_case[mask]
        return {
            "dataset_variant": dataset_variant,
            "model": model_name,
            "split": split_label,
            "n": len(sub),
            "sim_score_mean": float(sub["similarity_score"].mean()),
            "rouge_l_mean": float(sub["rouge_l"].mean()),
            "chrf_mean": float(sub["chrf"].mean()),
            "token_f1_mean": float(sub["token_f1"].mean()),
            "bertscore_f1_mean": float(sub["bertscore_f1"].mean()),
        }

    summaries = [_summary_for(per_case["split"] == s, s) for s in per_case["split"].unique()]
    summaries.append(_summary_for(pd.Series([True] * len(per_case)), "all"))

    return per_case, summaries


# ── main ───────────────────────────────────────────────────────────────────────

def collect_result_files(results_dir: Path) -> list[Path]:
    return sorted(p for p in results_dir.glob("*.json") if not p.stem.startswith("."))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evalúa baselines de retrieval")
    p.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default=None,
        help="Evalúa todos los modelos de ese dataset variant",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Evalúa un directorio concreto de resultados",
    )
    return p.parse_args()


def resolve_jobs(args: argparse.Namespace) -> list[tuple[Path, str, str]]:
    """Retorna lista de (results_file, model_name, dataset_variant)."""
    jobs: list[tuple[Path, str, str]] = []

    if args.results_dir:
        variant = args.results_dir.parts[-2].replace("dataset_", "")
        for f in collect_result_files(args.results_dir):
            model_name = f"{args.results_dir.name}/{f.stem}"
            jobs.append((f, model_name, variant))
        return jobs

    if args.dataset:
        dataset_dir = RESULTS_ROOT / f"dataset_{args.dataset}"
        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for f in collect_result_files(model_dir):
                model_name = f"{model_dir.name}/{f.stem}"
                jobs.append((f, model_name, args.dataset))
        return jobs

    # Sin argumentos: evalúa todo
    for variant in DATASETS:
        dataset_dir = RESULTS_ROOT / f"dataset_{variant}"
        if not dataset_dir.exists():
            continue
        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for f in collect_result_files(model_dir):
                model_name = f"{model_dir.name}/{f.stem}"
                jobs.append((f, model_name, variant))
    return jobs


def main() -> None:
    args = parse_args()
    jobs = resolve_jobs(args)

    if not jobs:
        print("No se encontraron archivos de resultados.")
        return

    # Agrupa por dataset_variant para cargar el dataset una sola vez
    by_variant: dict[str, list[tuple[Path, str]]] = {}
    for f, model_name, variant in jobs:
        by_variant.setdefault(variant, []).append((f, model_name))

    all_per_case: list[pd.DataFrame] = []
    all_summaries: list[dict[str, float]] = []

    for variant, variant_jobs in by_variant.items():
        dataset_path = DATASETS[variant]
        if not dataset_path.exists():
            print(f"Dataset no encontrado: {dataset_path} — saltando {variant}")
            continue

        records = load_dataset(dataset_path)
        reference_map = build_reference_map(records)
        split_map = build_split_map(records)
        print(f"\n=== dataset_{variant} ({len(records)} referencias) ===")

        for results_path, model_name in variant_jobs:
            per_case, summaries = evaluate_file(
                results_path, reference_map, split_map, model_name, variant
            )
            all_per_case.append(per_case)
            all_summaries.extend(summaries)

        # Guarda métricas por dataset
        out_dir = METRICS_ROOT / f"dataset_{variant}"
        out_dir.mkdir(parents=True, exist_ok=True)

        variant_per_case = pd.concat(
            [df for df in all_per_case if df["dataset_variant"].iloc[0] == variant],
            ignore_index=True,
        )
        variant_summaries = [s for s in all_summaries if s["dataset_variant"] == variant]

        variant_per_case.to_csv(out_dir / "metrics_per_case.csv", index=False)
        pd.DataFrame(variant_summaries).to_csv(out_dir / "metrics_summary.csv", index=False)
        print(f"\nGuardado en {out_dir}")

    # Tabla resumen en consola
    if all_summaries:
        df_summary = pd.DataFrame(all_summaries)
        cols = ["dataset_variant", "model", "split", "n", "rouge_l_mean", "chrf_mean",
                "token_f1_mean", "bertscore_f1_mean", "sim_score_mean"]
        print("\n" + "=" * 80)
        print("RESUMEN FINAL")
        print("=" * 80)
        print(df_summary[cols].to_string(index=False, float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
