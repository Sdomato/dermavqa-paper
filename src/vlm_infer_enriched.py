"""
Inference VLM sobre dataset_enriched.

Wrapper fino sobre `src.vlm_by_image_utils` para mantener el mismo contrato de
entrada/salida que `dataset_longest_answer_by_image`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.vlm_by_image_utils import (
    DATASETS_DIR,
    MODEL_ID,
    PROJECT_ROOT,
    ByImageDatasetConfig,
    build_chat_messages,
    build_inference_items as _build_inference_items,
    filter_split as _filter_split,
    load_by_image_dataset,
    run_by_image_inference,
)

DATASET_JSONL_PATH = DATASETS_DIR / "dermavqa_iiyi_llm_synthesized_answer_finetune.jsonl"
DATASET_CSV_PATH = DATASETS_DIR / "dermavqa_iiyi_llm_synthesized_answer_finetune.csv"
DATASET_ZIP_PATH = DATASETS_DIR / "dermavqa_iiyi_llm_synthesized_answer_finetune.zip"

RESULTS_ROOT = PROJECT_ROOT / "outputs" / "results" / "dataset_enriched"
DATASET_VARIANT = "dataset_enriched"
ANSWER_COLUMNS = ("synthesized_answer_es", "answer_es")
SPLIT_ALIASES = {"train": "train", "valid": "valid", "test": "test"}

CONFIG = ByImageDatasetConfig(
    dataset_name="dataset_enriched",
    dataset_variant=DATASET_VARIANT,
    default_paths=(DATASET_JSONL_PATH, DATASET_CSV_PATH, DATASET_ZIP_PATH),
    results_root=RESULTS_ROOT,
    answer_columns=ANSWER_COLUMNS,
    lora_method="vlm_lora",
    zero_shot_method="vlm_zero_shot",
    split_aliases=SPLIT_ALIASES,
    missing_message=(
        "No encontre el dataset enriched. Esperaba uno de: "
        f"{DATASET_JSONL_PATH}, {DATASET_CSV_PATH}, {DATASET_ZIP_PATH}"
    ),
)


def load_enriched_dataset(path: Path | None = None) -> list[dict[str, Any]]:
    return load_by_image_dataset(path, CONFIG.default_paths, CONFIG.missing_message)


def filter_split(records: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return _filter_split(records, split, SPLIT_ALIASES, CONFIG.dataset_name)


def build_inference_items(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _build_inference_items(records, ANSWER_COLUMNS)


def run(args: argparse.Namespace) -> None:
    run_by_image_inference(args, CONFIG)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inferencia VLM sobre dataset_enriched")
    parser.add_argument("--split", choices=list(SPLIT_ALIASES.keys()), default="valid")
    parser.add_argument("--dataset", type=Path, default=None, help="JSONL/CSV/ZIP enriched opcional")
    parser.add_argument("--model", default=MODEL_ID, help=f"Modelo HF (default: {MODEL_ID})")
    parser.add_argument("--adapter", default=None, help="Ruta a adapter LoRA")
    parser.add_argument("--quantize", choices=["4bit", "8bit"], default="4bit")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="Procesar solo las primeras N filas")
    parser.add_argument("--dry-run", action="store_true", help="Valida dataset, prompts e imagenes en CPU")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
