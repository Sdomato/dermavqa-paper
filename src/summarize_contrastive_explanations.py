"""Summarize structured explanations and prepare a blinded review sheet."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.vlm_by_image_utils import PROJECT_ROOT


METHODS_BY_DATASET = {
    "enriched": (
        "zero_shot",
        "zero_shot_rag",
        "lora",
        "lora_rag",
        "lora_rag_aware",
    ),
    "longest": ("zero_shot", "zero_shot_rag", "lora", "lora_rag"),
}
DATASET_NAMES = {
    "enriched": "dataset_enriched",
    "longest": "dataset_longest_answer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume explicaciones y crea planilla de revision."
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_NAMES),
        default="enriched",
    )
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--output-prefix",
        default="contrastive_explanation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_name = DATASET_NAMES[args.dataset]
    output_dir = args.output_dir or (
        PROJECT_ROOT / "outputs" / "error_analysis" / dataset_name
    )
    input_dir = args.input_dir or output_dir / "explanations"
    frames: list[pd.DataFrame] = []
    for method in METHODS_BY_DATASET[args.dataset]:
        path = input_dir / f"explanations_{method}.csv"
        if path.exists():
            frames.append(pd.read_csv(path).fillna(""))
        else:
            print(f"[warn] Falta {path}")
    if not frames:
        raise FileNotFoundError(
            f"No hay explicaciones en {input_dir}. Ejecuta primero "
            "src.vlm_explain_contrastive_cases."
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["parse_ok"] = combined["parse_error"].astype(str).str.len().eq(0)
    combined["meets_minimum_tokens"] = (
        combined["meets_minimum_tokens"]
        .astype(str)
        .str.lower()
        .isin({"true", "1"})
    )
    summary = (
        combined.groupby("method", as_index=False)
        .agg(
            n=("image_id", "count"),
            parse_success_rate=("parse_ok", "mean"),
            minimum_tokens_rate=("meets_minimum_tokens", "mean"),
            mean_explanation_tokens=("explanation_token_count", "mean"),
            mean_answer_stability_f1=(
                "original_vs_reanalysis_token_f1",
                "mean",
            ),
        )
        .sort_values("method")
    )

    review_columns = [
        "dataset_variant",
        "encounter_id",
        "image_id",
        "selection_buckets",
        "method",
        "question_es",
        "reference_answer_es",
        "original_prediction_es",
        "reanalysis_answer_es",
        "explanation",
        "visual_evidence_es",
        "question_evidence_es",
        "uncertainty_es",
        "rag_context_use_es",
        "original_vs_reanalysis_token_f1",
    ]
    for optional_column in ("explanation_mode", "candidate_support"):
        if optional_column in combined.columns:
            review_columns.insert(4, optional_column)
    review = combined[review_columns].copy()
    review["clinical_correctness_0_2"] = ""
    review["evidence_grounding_0_2"] = ""
    review["answer_explanation_consistency_0_2"] = ""
    review["hallucination_yes_no"] = ""
    review["rag_helped_hurt_neutral_na"] = ""
    review["error_category"] = ""
    review["reviewer_notes"] = ""

    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(
        output_dir / f"{args.output_prefix}s_all.csv",
        index=False,
        encoding="utf-8",
    )
    summary.to_csv(
        output_dir / f"{args.output_prefix}_summary.csv",
        index=False,
        encoding="utf-8",
    )
    review.to_csv(
        output_dir / f"{args.output_prefix}_review.csv",
        index=False,
        encoding="utf-8",
    )

    markdown = [
        "# Contrastive explanation run summary",
        "",
        "These explanations are post-hoc observable justifications, not hidden chain-of-thought traces.",
        "",
        "| method | n | parse success | minimum length | mean tokens | answer stability F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.to_dict(orient="records"):
        markdown.append(
            "| "
            f"{row['method']} | {row['n']} | "
            f"{row['parse_success_rate']:.3f} | "
            f"{row['minimum_tokens_rate']:.3f} | "
            f"{row['mean_explanation_tokens']:.1f} | "
            f"{row['mean_answer_stability_f1']:.3f} |"
        )
    (output_dir / f"{args.output_prefix}_summary.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(
        "Planilla de revision:",
        output_dir / f"{args.output_prefix}_review.csv",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
