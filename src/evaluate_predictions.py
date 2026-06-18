"""
Evaluación de predicciones VLM (zero-shot / LoRA) sobre dataset_longest_answer.

Toma uno o más CSV de predicciones con el esquema canónico del plan de equipo
(columnas predicted_answer_es y reference_answer_es) y calcula las MISMAS
métricas que usa src/evaluate_retrieval.py para los baselines de retrieval, de
modo que VLM zero-shot, VLM LoRA y retrieval multimodal sean directamente
comparables.

Métricas por caso:
  - chrF        (sacrebleu)          <- métrica automática principal para español
  - ROUGE-L F1  (rouge_score)
  - token-F1    (overlap de tokens normalizados)
  - BERTScore F1 multilingüe         (opcional, --no-bertscore para saltar)
  - cosine sim. semántica con E5     (opcional, --cosine)

Métricas a nivel corpus (en el summary):
  - sacreBLEU corpus
  - chrF corpus

Las fórmulas léxicas y de BERTScore replican exactamente las de
evaluate_retrieval.py (mismo modelo bert-base-multilingual-cased, mismo
use_stemmer=False, misma normalización) para garantizar comparabilidad. Se
inlinean acá en vez de importarlas porque evaluate_retrieval.py importa
bert_score/torch a nivel de módulo, lo que impediría correr solo las métricas
léxicas en CPU.

Salida:
  outputs/metrics/dataset_longest_answer/metrics_<split>.csv       (una fila por método)
  outputs/metrics/dataset_longest_answer/per_case_<method>_<split>.csv

Uso:
    # Evaluar un CSV de predicciones
    python -m src.evaluate_predictions \
        outputs/results/dataset_longest_answer/vlm_zero_shot/predictions_valid.csv

    # Evaluar varios y armar tabla comparativa (mismo split)
    python -m src.evaluate_predictions \
        outputs/results/dataset_longest_answer/vlm_zero_shot/predictions_test.csv \
        outputs/results/dataset_longest_answer/vlm_lora/predictions_test.csv

    # Solo métricas léxicas (sin torch), útil para validar en local
    python -m src.evaluate_predictions <csv> --no-bertscore
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.retrieval_utils import PROJECT_ROOT

METRICS_ROOT = PROJECT_ROOT / "outputs" / "metrics" / "dataset_longest_answer"
BERTSCORE_MODEL = "bert-base-multilingual-cased"
E5_MODEL_ID = "intfloat/multilingual-e5-base"


# ── métricas léxicas (idénticas a evaluate_retrieval.py) ─────────────────────────

def normalize(text: str) -> str:
    return " ".join(str(text).lower().split())


def token_f1(pred: str, ref: str) -> float:
    pred_tokens = normalize(pred).split()
    ref_tokens = normalize(ref).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    p = len(common) / len(pred_tokens)
    r = len(common) / len(ref_tokens)
    return 2 * p * r / (p + r)


def compute_rouge_l(predictions: list[str], references: list[str]) -> list[float]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for pred, ref in zip(predictions, references)
    ]


def compute_chrf(predictions: list[str], references: list[str]) -> list[float]:
    import sacrebleu

    return [
        sacrebleu.corpus_chrf([pred], [[ref]]).score / 100.0
        for pred, ref in zip(predictions, references)
    ]


def compute_bertscore(predictions: list[str], references: list[str]) -> list[float]:
    from bert_score import score as bert_score

    print(f"  Calculando BERTScore ({BERTSCORE_MODEL})...")
    _, _, f1 = bert_score(
        predictions, references, model_type=BERTSCORE_MODEL, lang="es", verbose=False
    )
    return f1.tolist()


def compute_cosine_e5(predictions: list[str], references: list[str]) -> list[float]:
    """Cosine entre embeddings E5 de predicción y referencia (semántica)."""
    from sentence_transformers import SentenceTransformer

    print(f"  Calculando cosine similarity ({E5_MODEL_ID})...")
    model = SentenceTransformer(E5_MODEL_ID)
    emb_pred = model.encode([f"query: {p}" for p in predictions], normalize_embeddings=True)
    emb_ref = model.encode([f"query: {r}" for r in references], normalize_embeddings=True)
    return [float(np.dot(a, b)) for a, b in zip(emb_pred, emb_ref)]


def corpus_scores(predictions: list[str], references: list[str]) -> dict[str, float]:
    import sacrebleu

    bleu = sacrebleu.corpus_bleu(predictions, [references]).score
    chrf = sacrebleu.corpus_chrf(predictions, [references]).score
    return {"sacrebleu_corpus": bleu, "chrf_corpus": chrf}


# ── evaluación de un CSV ─────────────────────────────────────────────────────────

def evaluate_csv(
    csv_path: Path, use_bertscore: bool, use_cosine: bool
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(csv_path).fillna({"predicted_answer_es": "", "reference_answer_es": ""})

    method = str(df["method"].iloc[0]) if "method" in df else csv_path.stem
    model_name = str(df["model_name"].iloc[0]) if "model_name" in df else "unknown"
    split = str(df["split"].iloc[0]) if "split" in df else "unknown"

    preds = df["predicted_answer_es"].astype(str).tolist()
    refs = df["reference_answer_es"].astype(str).tolist()
    n_empty = sum(1 for p in preds if not p.strip())

    print(f"\n[{method}] {len(preds)} casos ({n_empty} predicciones vacías)")

    chrf = compute_chrf(preds, refs)
    rouge_l = compute_rouge_l(preds, refs)
    tok_f1 = [token_f1(p, r) for p, r in zip(preds, refs)]

    per_case = pd.DataFrame(
        {
            "method": method,
            "model_name": model_name,
            "split": split,
            "encounter_id": df.get("encounter_id", pd.Series(range(len(df)))),
            "chrf": chrf,
            "rouge_l": rouge_l,
            "token_f1": tok_f1,
        }
    )

    summary: dict[str, Any] = {
        "method": method,
        "model_name": model_name,
        "split": split,
        "n": len(preds),
        "empty_predictions": n_empty,
        "chrf_mean": float(np.mean(chrf)),
        "rouge_l_mean": float(np.mean(rouge_l)),
        "token_f1_mean": float(np.mean(tok_f1)),
    }
    summary.update(corpus_scores(preds, refs))

    if use_bertscore:
        bscore = compute_bertscore(preds, refs)
        per_case["bertscore_f1"] = bscore
        summary["bertscore_f1_mean"] = float(np.mean(bscore))

    if use_cosine:
        cos = compute_cosine_e5(preds, refs)
        per_case["cosine_e5"] = cos
        summary["cosine_e5_mean"] = float(np.mean(cos))

    return per_case, summary


# ── main ─────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    METRICS_ROOT.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    split_seen: set[str] = set()

    for csv_path in args.predictions:
        per_case, summary = evaluate_csv(
            Path(csv_path), use_bertscore=not args.no_bertscore, use_cosine=args.cosine
        )
        summaries.append(summary)
        split_seen.add(summary["split"])

        pc_path = METRICS_ROOT / f"per_case_{summary['method']}_{summary['split']}.csv"
        per_case.to_csv(pc_path, index=False, encoding="utf-8")
        print(f"  Por caso -> {pc_path}")

    df_summary = pd.DataFrame(summaries)

    # Si todos comparten split, escribir metrics_<split>.csv (naming del plan de equipo).
    # Mezcla con métricas previas del mismo split sin duplicar método.
    if len(split_seen) == 1:
        split = next(iter(split_seen))
        out_path = METRICS_ROOT / f"metrics_{split}.csv"
        if out_path.exists():
            prev = pd.read_csv(out_path)
            prev = prev[~prev["method"].isin(df_summary["method"])]
            df_summary = pd.concat([prev, df_summary], ignore_index=True)
        df_summary.to_csv(out_path, index=False, encoding="utf-8")
        print(f"\nResumen -> {out_path}")
    else:
        out_path = METRICS_ROOT / "metrics_mixed.csv"
        df_summary.to_csv(out_path, index=False, encoding="utf-8")
        print(f"\nResumen (splits mixtos) -> {out_path}")

    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(df_summary.to_string(index=False, float_format="{:.4f}".format))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evalúa predicciones VLM sobre longest_answer")
    p.add_argument("predictions", nargs="+", help="Uno o más CSV de predicciones")
    p.add_argument("--no-bertscore", action="store_true", help="Saltear BERTScore (sin torch)")
    p.add_argument("--cosine", action="store_true", help="Calcular cosine semántica con E5")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
