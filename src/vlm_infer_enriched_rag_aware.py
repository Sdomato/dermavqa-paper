"""
Inferencia VLM sobre dataset_enriched con prompt RAG-aware pre-computado.

Este wrapper evalua el adapter entrenado con `src.train_enriched_rag` usando el
mismo formato de prompt de entrenamiento: imagen actual + pregunta + contextos
RAG recuperados desde train.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.vlm_by_image_utils import (
    DATASETS_DIR,
    MODEL_ID,
    PROJECT_ROOT,
    ByImageDatasetConfig,
    run_rag_by_image_inference,
)
from src.vlm_infer_enriched import SPLIT_ALIASES

RAG_DATASET_PATH = DATASETS_DIR / "dermavqa_iiyi_llm_synthesized_answer_finetune_rag_e5_small.jsonl"
RESULTS_ROOT = PROJECT_ROOT / "outputs" / "results" / "dataset_enriched"

CONFIG = ByImageDatasetConfig(
    dataset_name="dataset_enriched_rag_aware",
    dataset_variant="dataset_enriched",
    default_paths=(RAG_DATASET_PATH,),
    results_root=RESULTS_ROOT,
    answer_columns=("answer_es",),
    lora_method="vlm_lora_rag_aware",
    zero_shot_method="vlm_zero_shot_rag_aware",
    split_aliases=SPLIT_ALIASES,
    missing_message=(
        "No encontre el dataset enriched RAG-aware. Ejecuta primero: "
        "python -m src.build_rag_training_dataset --dataset enriched --top-k 3"
    ),
)


def run(args: argparse.Namespace) -> None:
    run_rag_by_image_inference(args, CONFIG)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inferencia VLM RAG-aware sobre dataset_enriched")
    parser.add_argument("--split", choices=list(SPLIT_ALIASES.keys()), default="valid")
    parser.add_argument("--dataset", type=Path, default=None, help="JSONL enriched RAG-aware opcional")
    parser.add_argument("--model", default=MODEL_ID, help=f"Modelo HF (default: {MODEL_ID})")
    parser.add_argument("--adapter", default=None, help="Ruta a adapter LoRA")
    parser.add_argument("--quantize", choices=["4bit", "8bit"], default="4bit")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="Procesar solo las primeras N filas")
    parser.add_argument("--dry-run", action="store_true", help="Valida dataset, prompts e imagenes en CPU")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
