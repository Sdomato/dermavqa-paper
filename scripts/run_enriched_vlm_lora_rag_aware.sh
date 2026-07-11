#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
TOP_K="${TOP_K:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
DATASET="outputs/datasets/dermavqa_iiyi_llm_synthesized_answer_finetune_rag_e5_small.jsonl"
LOG_DIR="outputs/results/dataset_enriched/vlm_lora_rag_aware"
ADAPTER="$LOG_DIR/final_adapter"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$LOG_DIR"

printf "\n[1/6] Building enriched RAG-aware dataset (top-k=%s)...\n" "$TOP_K"
"$PYTHON_BIN" -m src.build_rag_training_dataset \
  --dataset enriched \
  --top-k "$TOP_K" \
  --out "$DATASET" \
  2>&1 | tee "$LOG_DIR/build_rag_dataset.log"

printf "\n[2/6] Training dataset_enriched RAG-aware QLoRA...\n"
"$PYTHON_BIN" -m src.train_enriched_rag "$@" \
  2>&1 | tee "$LOG_DIR/train_enriched_rag.log"

printf "\n[3/6] Generating valid predictions with RAG-aware prompt...\n"
"$PYTHON_BIN" -m src.vlm_infer_enriched_rag_aware \
  --split valid \
  --dataset "$DATASET" \
  --adapter "$ADAPTER" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  2>&1 | tee "$LOG_DIR/infer_valid.log"

printf "\n[4/6] Generating test predictions with RAG-aware prompt...\n"
"$PYTHON_BIN" -m src.vlm_infer_enriched_rag_aware \
  --split test \
  --dataset "$DATASET" \
  --adapter "$ADAPTER" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  2>&1 | tee "$LOG_DIR/infer_test.log"

printf "\n[5/6] Computing valid/test metrics...\n"
"$PYTHON_BIN" -m src.evaluate_predictions \
  "$LOG_DIR/predictions_valid.csv" \
  "$LOG_DIR/predictions_test.csv" \
  2>&1 | tee "$LOG_DIR/evaluate_predictions.log"

printf "\n[6/6] Done. Main artifacts:\n"
printf "  Adapter: %s\n" "$ADAPTER"
printf "  Train runtime: %s\n" "$LOG_DIR/train_runtime.json"
printf "  Train metrics: %s\n" "$LOG_DIR/train_metrics.json"
printf "  Valid eval loss metrics: %s\n" "$LOG_DIR/eval_metrics_valid.json"
printf "  Predictions: %s/predictions_valid.csv and predictions_test.csv\n" "$LOG_DIR"
printf "  Generation metrics: outputs/metrics/dataset_enriched/metrics_mixed.csv\n"
