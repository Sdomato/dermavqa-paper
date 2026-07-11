"""Build a contrastive sample for qualitative VLM error analysis.

The script joins per-case metrics and predictions from the four enriched
dataset inference conditions:

* base VLM;
* base VLM with RAG;
* LoRA VLM;
* LoRA VLM with RAG at inference.

It selects cases with large pairwise metric changes and writes both the
selected cases and aggregate metric summaries by observable strata.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.vlm_by_image_utils import PROJECT_ROOT, clean_text


METRICS = ("chrf", "rouge_l", "token_f1", "bertscore_f1")


@dataclass(frozen=True)
class MethodFiles:
    alias: str
    metrics: Path
    predictions: Path


@dataclass(frozen=True)
class DatasetExperiment:
    name: str
    methods: tuple[MethodFiles, ...]

    @property
    def output_dir(self) -> Path:
        return PROJECT_ROOT / "outputs" / "error_analysis" / self.name


DATASET_EXPERIMENTS = {
    "enriched": DatasetExperiment(
        name="dataset_enriched",
        methods=(
            MethodFiles(
                alias="zero_shot",
                metrics=PROJECT_ROOT
                / "outputs/metrics/dataset_enriched/"
                "per_case_vlm_zero_shot_test.csv",
                predictions=PROJECT_ROOT
                / "outputs/results/dataset_enriched/"
                "vlm_zero_shot/predictions_test.csv",
            ),
            MethodFiles(
                alias="zero_shot_rag",
                metrics=PROJECT_ROOT
                / "outputs/metrics/dataset_enriched/"
                "per_case_vlm_zero_shot_rag_e5_small_enriched_test.csv",
                predictions=PROJECT_ROOT
                / "outputs/results/dataset_enriched/"
                "vlm_zero_shot_rag_e5_small_enriched/predictions_test.csv",
            ),
            MethodFiles(
                alias="lora",
                metrics=PROJECT_ROOT
                / "outputs/metrics/dataset_enriched/"
                "per_case_vlm_lora_test.csv",
                predictions=PROJECT_ROOT
                / "outputs/results/dataset_enriched/"
                "vlm_lora/predictions_test.csv",
            ),
            MethodFiles(
                alias="lora_rag",
                metrics=PROJECT_ROOT
                / "outputs/metrics/dataset_enriched/"
                "per_case_vlm_lora_rag_e5_small_enriched_test.csv",
                predictions=PROJECT_ROOT
                / "outputs/results/dataset_enriched/"
                "vlm_lora_rag_e5_small_enriched/predictions_test.csv",
            ),
        ),
    ),
    "longest": DatasetExperiment(
        name="dataset_longest_answer",
        methods=(
            MethodFiles(
                alias="zero_shot",
                metrics=PROJECT_ROOT
                / "outputs/metrics/dataset_longest_answer/"
                "per_case_vlm_zero_shot_by_image_test.csv",
                predictions=PROJECT_ROOT
                / "outputs/results/dataset_longest_answer/"
                "vlm_zero_shot_by_image/predictions_test.csv",
            ),
            MethodFiles(
                alias="zero_shot_rag",
                metrics=PROJECT_ROOT
                / "outputs/metrics/dataset_longest_answer/"
                "per_case_vlm_zero_shot_by_image_rag_e5_small_longest_test.csv",
                predictions=PROJECT_ROOT
                / "outputs/results/dataset_longest_answer/"
                "vlm_zero_shot_by_image_rag_e5_small_longest/"
                "predictions_test.csv",
            ),
            MethodFiles(
                alias="lora",
                metrics=PROJECT_ROOT
                / "outputs/metrics/dataset_longest_answer/"
                "per_case_vlm_lora_by_image_test.csv",
                predictions=PROJECT_ROOT
                / "outputs/results/dataset_longest_answer/"
                "vlm_lora_by_image/predictions_test.csv",
            ),
            MethodFiles(
                alias="lora_rag",
                metrics=PROJECT_ROOT
                / "outputs/metrics/dataset_longest_answer/"
                "per_case_vlm_lora_by_image_rag_e5_small_longest_test.csv",
                predictions=PROJECT_ROOT
                / "outputs/results/dataset_longest_answer/"
                "vlm_lora_by_image_rag_e5_small_longest/predictions_test.csv",
            ),
        ),
    ),
}


def word_count(value: object) -> int:
    return len(clean_text(value).split())


def require_files(methods: tuple[MethodFiles, ...]) -> None:
    missing = [
        path
        for method in methods
        for path in (method.metrics, method.predictions)
        if not path.exists()
    ]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Faltan resultados requeridos:\n{formatted}")


def load_wide_table(experiment: DatasetExperiment) -> pd.DataFrame:
    methods = experiment.methods
    require_files(methods)
    key = ["encounter_id", "image_id"]
    wide: pd.DataFrame | None = None

    for method in methods:
        metrics = pd.read_csv(method.metrics)
        metrics = metrics[key + list(METRICS)].copy()
        metrics = metrics.rename(
            columns={metric: f"{method.alias}_{metric}" for metric in METRICS}
        )

        predictions = pd.read_csv(method.predictions).fillna("")
        prediction_columns = key + ["predicted_answer_es"]
        predictions = predictions[prediction_columns].rename(
            columns={
                "predicted_answer_es": f"{method.alias}_predicted_answer_es"
            }
        )
        current = metrics.merge(predictions, on=key, how="inner", validate="one_to_one")
        wide = (
            current
            if wide is None
            else wide.merge(current, on=key, how="inner", validate="one_to_one")
        )

    base_predictions = pd.read_csv(methods[0].predictions).fillna("")
    base_columns = key + [
        "split",
        "image_path",
        "question_es",
        "reference_answer_es",
    ]
    wide = wide.merge(
        base_predictions[base_columns],
        on=key,
        how="left",
        validate="one_to_one",
    )

    rag_predictions = pd.read_csv(methods[1].predictions).fillna("")
    rag_columns = [
        column
        for column in (
            "encounter_id",
            "image_id",
            "rag_context_es",
            "retrieved_encounter_ids",
            "retrieved_scores",
            "rag_retriever",
            "rag_top_k",
        )
        if column in rag_predictions.columns
    ]
    if len(rag_columns) > 2:
        wide = wide.merge(
            rag_predictions[rag_columns],
            on=key,
            how="left",
            validate="one_to_one",
        )

    for method in methods:
        metric_columns = [f"{method.alias}_{metric}" for metric in METRICS]
        wide[f"{method.alias}_mean_score"] = wide[metric_columns].mean(axis=1)

    wide["delta_lora_vs_zero_shot"] = (
        wide["lora_mean_score"] - wide["zero_shot_mean_score"]
    )
    wide["delta_rag_on_zero_shot"] = (
        wide["zero_shot_rag_mean_score"] - wide["zero_shot_mean_score"]
    )
    wide["delta_rag_on_lora"] = (
        wide["lora_rag_mean_score"] - wide["lora_mean_score"]
    )
    wide["question_words"] = wide["question_es"].map(word_count)
    wide["reference_words"] = wide["reference_answer_es"].map(word_count)
    wide["images_in_encounter"] = wide.groupby("encounter_id")["image_id"].transform(
        "count"
    )
    wide["lora_image_spread"] = wide.groupby("encounter_id")[
        "lora_mean_score"
    ].transform(lambda values: float(values.max() - values.min()))

    lexical = wide[
        ["lora_chrf", "lora_rouge_l", "lora_token_f1"]
    ].mean(axis=1)
    lexical_std = lexical.std(ddof=0) or 1.0
    semantic_std = wide["lora_bertscore_f1"].std(ddof=0) or 1.0
    lexical_z = (lexical - lexical.mean()) / lexical_std
    semantic_z = (
        wide["lora_bertscore_f1"] - wide["lora_bertscore_f1"].mean()
    ) / semantic_std
    wide["semantic_lexical_gap_z"] = semantic_z - lexical_z
    wide["dataset_variant"] = experiment.name
    return wide


def add_ranked(
    selections: dict[tuple[str, str], dict[str, object]],
    frame: pd.DataFrame,
    bucket: str,
    score_column: str,
    count: int,
    largest: bool,
) -> None:
    ranked = frame.sort_values(score_column, ascending=not largest).head(count)
    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        key = (str(row["encounter_id"]), str(row["image_id"]))
        entry = selections.setdefault(
            key,
            {
                "row": row,
                "buckets": [],
                "reasons": [],
            },
        )
        entry["buckets"].append(bucket)
        entry["reasons"].append(
            f"{bucket}: rank={rank}, {score_column}={row[score_column]:.4f}"
        )


def select_cases(wide: pd.DataFrame, n_per_bucket: int) -> pd.DataFrame:
    selections: dict[tuple[str, str], dict[str, object]] = {}
    ranked_specs = (
        ("lora_gain", "delta_lora_vs_zero_shot", True),
        ("lora_harm", "delta_lora_vs_zero_shot", False),
        ("rag_gain_zero_shot", "delta_rag_on_zero_shot", True),
        ("rag_harm_zero_shot", "delta_rag_on_zero_shot", False),
        ("rag_gain_lora", "delta_rag_on_lora", True),
        ("rag_harm_lora", "delta_rag_on_lora", False),
        ("semantic_high_lexical_low", "semantic_lexical_gap_z", True),
        ("semantic_low_lexical_high", "semantic_lexical_gap_z", False),
        ("long_reference", "reference_words", True),
        ("short_reference", "reference_words", False),
    )
    for bucket, column, largest in ranked_specs:
        add_ranked(
            selections,
            wide,
            bucket=bucket,
            score_column=column,
            count=n_per_bucket,
            largest=largest,
        )

    multi_image = wide[wide["images_in_encounter"] > 1]
    top_encounters = (
        multi_image[["encounter_id", "lora_image_spread"]]
        .drop_duplicates()
        .sort_values("lora_image_spread", ascending=False)
        .head(n_per_bucket)["encounter_id"]
    )
    for encounter_id in top_encounters:
        encounter_rows = multi_image[multi_image["encounter_id"] == encounter_id]
        add_ranked(
            selections,
            encounter_rows,
            bucket="multi_image_inconsistency",
            score_column="lora_image_spread",
            count=len(encounter_rows),
            largest=True,
        )

    rows: list[dict[str, object]] = []
    for entry in selections.values():
        row = entry["row"].to_dict()
        row["selection_buckets"] = ";".join(entry["buckets"])
        row["selection_reasons"] = " | ".join(entry["reasons"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["selection_buckets", "encounter_id", "image_id"]
    )


def length_bucket(words: int) -> str:
    if words <= 20:
        return "short_0_20"
    if words <= 60:
        return "medium_21_60"
    return "long_61_plus"


def build_strata_summary(
    wide: pd.DataFrame, methods: tuple[MethodFiles, ...]
) -> pd.DataFrame:
    frame = wide.copy()
    frame["reference_length"] = frame["reference_words"].map(length_bucket)
    frame["question_length"] = frame["question_words"].map(length_bucket)
    frame["image_count"] = np.where(
        frame["images_in_encounter"] > 1, "multiple_images", "single_image"
    )

    rows: list[dict[str, object]] = []
    for dimension in ("reference_length", "question_length", "image_count"):
        for stratum, group in frame.groupby(dimension):
            for method in methods:
                row: dict[str, object] = {
                    "dimension": dimension,
                    "stratum": stratum,
                    "method": method.alias,
                    "n": len(group),
                }
                for metric in METRICS:
                    row[metric] = group[f"{method.alias}_{metric}"].mean()
                row["mean_score"] = group[f"{method.alias}_mean_score"].mean()
                rows.append(row)
    return pd.DataFrame(rows)


def build_effect_summary(wide: pd.DataFrame) -> pd.DataFrame:
    effects = (
        ("lora_vs_zero_shot", "delta_lora_vs_zero_shot"),
        ("rag_on_zero_shot", "delta_rag_on_zero_shot"),
        ("rag_on_lora", "delta_rag_on_lora"),
    )
    frame = wide.copy()
    frame["reference_length"] = frame["reference_words"].map(length_bucket)
    frame["image_count"] = np.where(
        frame["images_in_encounter"] > 1, "multiple_images", "single_image"
    )
    rows: list[dict[str, object]] = []
    for effect_name, column in effects:
        rows.append(
            {
                "effect": effect_name,
                "dimension": "overall",
                "stratum": "all",
                "n": len(frame),
                "mean_delta": frame[column].mean(),
                "median_delta": frame[column].median(),
                "positive_rate": (frame[column] > 0).mean(),
            }
        )
        for dimension in ("reference_length", "image_count"):
            for stratum, group in frame.groupby(dimension):
                rows.append(
                    {
                        "effect": effect_name,
                        "dimension": dimension,
                        "stratum": stratum,
                        "n": len(group),
                        "mean_delta": group[column].mean(),
                        "median_delta": group[column].median(),
                        "positive_rate": (group[column] > 0).mean(),
                    }
                )
    return pd.DataFrame(rows)


def write_jsonl(frame: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in frame.replace({np.nan: ""}).to_dict(orient="records"):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Selecciona casos contrastivos para explicar resultados VLM."
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_EXPERIMENTS),
        default="enriched",
    )
    parser.add_argument("--n-per-bucket", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.n_per_bucket < 1:
        raise ValueError("--n-per-bucket debe ser al menos 1")

    experiment = DATASET_EXPERIMENTS[args.dataset]
    output_dir = args.output_dir or experiment.output_dir
    wide = load_wide_table(experiment)
    selected = select_cases(wide, args.n_per_bucket)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_csv = output_dir / "contrastive_cases_test.csv"
    selected_jsonl = output_dir / "contrastive_cases_test.jsonl"
    strata_csv = output_dir / "metric_strata_summary.csv"
    effects_csv = output_dir / "pairwise_effect_summary.csv"

    selected.to_csv(selected_csv, index=False, encoding="utf-8")
    write_jsonl(selected, selected_jsonl)
    build_strata_summary(wide, experiment.methods).to_csv(
        strata_csv, index=False, encoding="utf-8"
    )
    build_effect_summary(wide).to_csv(effects_csv, index=False, encoding="utf-8")

    print(f"Dataset: {experiment.name}")
    print(f"Casos completos comparables: {len(wide)}")
    print(f"Casos contrastivos únicos: {len(selected)}")
    print(f"Casos: {selected_csv}")
    print(f"Estratos: {strata_csv}")
    print(f"Efectos pareados: {effects_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
