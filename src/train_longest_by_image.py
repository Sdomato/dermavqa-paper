"""
Fine-tuning LoRA/QLoRA de Qwen2.5-VL sobre dataset_longest_answer_by_image.

Wrapper reproducible sobre `src.vlm_lora_training`. Usa exactamente el mismo
motor que `src.train_enriched`; solo cambia el dataset y el target.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.vlm_infer_longest_by_image import (
    MODEL_ID,
    RESULTS_ROOT,
    build_chat_messages,
    build_inference_items,
    filter_split,
    load_longest_by_image_dataset,
)
from src.vlm_lora_training import VlmLoraTrainConfig, run_lora_training

RUN_DIR = RESULTS_ROOT / "vlm_lora_by_image"
ADAPTER_DIR = RUN_DIR / "final_adapter"
CHECKPOINTS_DIR = RUN_DIR / "checkpoints"
RUNTIME_PATH = RUN_DIR / "train_runtime.json"
TRAINING_CONFIG_PATH = RUN_DIR / "training_config.json"
TRAIN_METRICS_PATH = RUN_DIR / "train_metrics.json"
VALID_METRICS_PATH = RUN_DIR / "eval_metrics_valid.json"
TRAINER_STATE_PATH = RUN_DIR / "trainer_state.json"
LOG_HISTORY_JSON_PATH = RUN_DIR / "training_log_history.json"
LOG_HISTORY_CSV_PATH = RUN_DIR / "training_log_history.csv"

TRAIN_CONFIG = VlmLoraTrainConfig(
    dataset_variant="dataset_longest_answer",
    comparison_protocol="matched_to_dataset_enriched_by_image",
    training_unit="one_image_row",
    run_dir=RUN_DIR,
    adapter_dir=ADAPTER_DIR,
    checkpoints_dir=CHECKPOINTS_DIR,
    runtime_path=RUNTIME_PATH,
    training_config_path=TRAINING_CONFIG_PATH,
    train_metrics_path=TRAIN_METRICS_PATH,
    valid_metrics_path=VALID_METRICS_PATH,
    trainer_state_path=TRAINER_STATE_PATH,
    log_history_json_path=LOG_HISTORY_JSON_PATH,
    log_history_csv_path=LOG_HISTORY_CSV_PATH,
    load_records=load_longest_by_image_dataset,
    filter_split=filter_split,
    build_inference_items=build_inference_items,
    build_chat_messages=build_chat_messages,
)


def run(args: argparse.Namespace) -> None:
    run_lora_training(args, TRAIN_CONFIG)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tuning LoRA/QLoRA de Qwen2.5-VL sobre dataset_longest_answer_by_image"
    )
    parser.add_argument("--dataset", type=Path, default=None, help="JSON/JSONL/CSV/ZIP by-image opcional")
    parser.add_argument("--model", default=MODEL_ID, help=f"Modelo HF (default: {MODEL_ID})")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs de entrenamiento; default 1 para comparar con enriched.")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=0, help="Cantidad maxima de checkpoints a conservar; 0 conserva todos.")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Usar solo las primeras N filas por split")
    parser.add_argument("--dry-run", action="store_true", help="Valida formato chat e imagenes en CPU")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
