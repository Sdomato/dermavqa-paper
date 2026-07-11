"""
Fine-tuning LoRA/QLoRA de Qwen2.5-VL sobre dataset_longest_answer_by_image con RAG-aware.

El dataset de entrada debe ser el JSONL pre-computado por build_rag_training_dataset.py
(campo rag_contexts ya incluido). El prompt de entrenamiento es idéntico al que
se usa en inferencia RAG, cerrando el mismatch training/inferencia.

Paso previo obligatorio:
    python -m src.build_rag_training_dataset --dataset longest --top-k 3

Entrenamiento:
    python -m src.train_longest_by_image_rag [--epochs 1] [--dry-run]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.vlm_by_image_utils import (
    MODEL_ID,
    build_rag_chat_messages,
    build_rag_inference_items,
)
from src.vlm_infer_longest_by_image import RESULTS_ROOT, SPLIT_ALIASES, load_longest_by_image_dataset
from src.vlm_lora_training import VlmLoraTrainConfig, run_lora_training

RAG_DATASET_PATH = Path("outputs/datasets/dataset_longest_answer_by_image_rag_e5_small.jsonl")
ANSWER_COLUMNS = ("answer_es",)

RUN_DIR = RESULTS_ROOT / "vlm_lora_by_image_rag_aware"
ADAPTER_DIR = RUN_DIR / "final_adapter"
CHECKPOINTS_DIR = RUN_DIR / "checkpoints"

TRAIN_CONFIG = VlmLoraTrainConfig(
    dataset_variant="dataset_longest_answer",
    comparison_protocol="rag_aware_e5_small_top3",
    training_unit="one_image_row_with_rag_context",
    run_dir=RUN_DIR,
    adapter_dir=ADAPTER_DIR,
    checkpoints_dir=CHECKPOINTS_DIR,
    runtime_path=RUN_DIR / "train_runtime.json",
    training_config_path=RUN_DIR / "training_config.json",
    train_metrics_path=RUN_DIR / "train_metrics.json",
    valid_metrics_path=RUN_DIR / "eval_metrics_valid.json",
    trainer_state_path=RUN_DIR / "trainer_state.json",
    log_history_json_path=RUN_DIR / "training_log_history.json",
    log_history_csv_path=RUN_DIR / "training_log_history.csv",
    load_records=load_longest_by_image_dataset,
    filter_split=lambda records, split: [r for r in records if str(r.get("split", "")) == SPLIT_ALIASES.get(split, split)],
    build_inference_items=lambda records: build_rag_inference_items(records, ANSWER_COLUMNS),
    build_chat_messages=build_rag_chat_messages,
)


def run(args: argparse.Namespace) -> None:
    if args.dataset is None:
        args.dataset = RAG_DATASET_PATH
    run_lora_training(args, TRAIN_CONFIG)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tuning LoRA RAG-aware sobre dataset_longest_answer_by_image"
    )
    p.add_argument("--dataset", type=Path, default=None, help=f"JSONL RAG pre-computado (default: {RAG_DATASET_PATH})")
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--save-total-limit", type=int, default=0)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
