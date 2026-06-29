#!/usr/bin/env bash
set -euo pipefail

ADAPTER="outputs/results/dataset_enriched/vlm_lora/final_adapter"
LOG_DIR="outputs/results/dataset_enriched/vlm_lora"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$LOG_DIR"

printf "\n[1/5] Training dataset_enriched QLoRA...\n"
"$PYTHON_BIN" -m src.train_enriched "$@" 2>&1 | tee "$LOG_DIR/train_enriched.log"

printf "\n[2/5] Generating valid predictions...\n"
"$PYTHON_BIN" -m src.vlm_infer_enriched --split valid --adapter "$ADAPTER" 2>&1 | tee "$LOG_DIR/infer_valid.log"

printf "\n[3/5] Generating test predictions...\n"
"$PYTHON_BIN" -m src.vlm_infer_enriched --split test --adapter "$ADAPTER" 2>&1 | tee "$LOG_DIR/infer_test.log"

printf "\n[4/5] Computing valid/test metrics...\n"
"$PYTHON_BIN" -m src.evaluate_predictions   "$LOG_DIR/predictions_valid.csv"   "$LOG_DIR/predictions_test.csv" 2>&1 | tee "$LOG_DIR/evaluate_predictions.log"

printf "\n[5/5] Done. Main artifacts:\n"
printf "  Adapter: %s\n" "$ADAPTER"
printf "  Train runtime: %s\n" "$LOG_DIR/train_runtime.json"
printf "  Train metrics: %s\n" "$LOG_DIR/train_metrics.json"
printf "  Valid eval loss metrics: %s\n" "$LOG_DIR/eval_metrics_valid.json"
printf "  Predictions: %s/predictions_valid.csv and predictions_test.csv\n" "$LOG_DIR"
printf "  Generation metrics: outputs/metrics/dataset_enriched/metrics_mixed.csv\n"
