#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_VARIANT="${DATASET_VARIANT:-enriched}"
MIN_EXPLANATION_TOKENS="${MIN_EXPLANATION_TOKENS:-100}"
N_PER_BUCKET="${N_PER_BUCKET:-2}"

case "$DATASET_VARIANT" in
  enriched)
    DATASET_NAME="dataset_enriched"
    DEFAULT_ADAPTER="outputs/results/dataset_enriched/vlm_lora/final_adapter"
    DEFAULT_RAG_AWARE_ADAPTER="outputs/results/dataset_enriched/vlm_lora_rag_aware/final_adapter"
    ;;
  longest)
    DATASET_NAME="dataset_longest_answer"
    DEFAULT_ADAPTER="outputs/results/dataset_longest_answer/vlm_lora_by_image/final_adapter"
    ;;
  *)
    printf "[error] DATASET_VARIANT debe ser enriched o longest\n" >&2
    exit 1
    ;;
esac

ADAPTER="${ADAPTER:-$DEFAULT_ADAPTER}"
RAG_AWARE_ADAPTER="${RAG_AWARE_ADAPTER:-${DEFAULT_RAG_AWARE_ADAPTER:-}}"
OUTPUT_DIR="outputs/error_analysis/$DATASET_NAME"
LOG_DIR="$OUTPUT_DIR/logs"

mkdir -p "$LOG_DIR"

"$PYTHON_BIN" - "$OUTPUT_DIR/run_manifest.json" <<PY
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "dataset_variant": "$DATASET_VARIANT",
    "dataset_name": "$DATASET_NAME",
    "model": "Qwen/Qwen2.5-VL-3B-Instruct",
    "adapter": "$ADAPTER",
    "rag_aware_adapter": "$RAG_AWARE_ADAPTER",
    "minimum_explanation_tokens": int("$MIN_EXPLANATION_TOKENS"),
    "n_per_bucket": int("$N_PER_BUCKET"),
    "started_at": datetime.now(timezone.utc).isoformat(),
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "git_dirty": bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip()
    ),
}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

DRY_RUN_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN_ARGS=(--dry-run)
fi

printf "\n[1/7] Building contrastive sample and metric strata...\n"
"$PYTHON_BIN" -m src.build_contrastive_explanation_sample \
  --dataset "$DATASET_VARIANT" \
  --n-per-bucket "$N_PER_BUCKET" \
  2>&1 | tee "$LOG_DIR/build_sample.log"

printf "\n[2/7] Re-inference: base VLM...\n"
"$PYTHON_BIN" -m src.vlm_explain_contrastive_cases \
  --dataset "$DATASET_VARIANT" \
  --method zero_shot \
  --min-explanation-tokens "$MIN_EXPLANATION_TOKENS" \
  "${LIMIT_ARGS[@]}" \
  "${DRY_RUN_ARGS[@]}" \
  2>&1 | tee "$LOG_DIR/explain_zero_shot.log"

printf "\n[3/7] Re-inference: base VLM + RAG...\n"
"$PYTHON_BIN" -m src.vlm_explain_contrastive_cases \
  --dataset "$DATASET_VARIANT" \
  --method zero_shot_rag \
  --min-explanation-tokens "$MIN_EXPLANATION_TOKENS" \
  "${LIMIT_ARGS[@]}" \
  "${DRY_RUN_ARGS[@]}" \
  2>&1 | tee "$LOG_DIR/explain_zero_shot_rag.log"

if [[ ! -d "$ADAPTER" && "${DRY_RUN:-0}" != "1" ]]; then
  printf "[error] No existe el adapter LoRA: %s\n" "$ADAPTER" >&2
  exit 1
fi

printf "\n[4/7] Re-inference: LoRA VLM...\n"
"$PYTHON_BIN" -m src.vlm_explain_contrastive_cases \
  --dataset "$DATASET_VARIANT" \
  --method lora \
  --adapter "$ADAPTER" \
  --min-explanation-tokens "$MIN_EXPLANATION_TOKENS" \
  "${LIMIT_ARGS[@]}" \
  "${DRY_RUN_ARGS[@]}" \
  2>&1 | tee "$LOG_DIR/explain_lora.log"

printf "\n[5/7] Re-inference: LoRA VLM + RAG...\n"
"$PYTHON_BIN" -m src.vlm_explain_contrastive_cases \
  --dataset "$DATASET_VARIANT" \
  --method lora_rag \
  --adapter "$ADAPTER" \
  --min-explanation-tokens "$MIN_EXPLANATION_TOKENS" \
  "${LIMIT_ARGS[@]}" \
  "${DRY_RUN_ARGS[@]}" \
  2>&1 | tee "$LOG_DIR/explain_lora_rag.log"

if [[ "$DATASET_VARIANT" == "enriched" ]]; then
  if [[ ! -d "$RAG_AWARE_ADAPTER" && "${DRY_RUN:-0}" != "1" ]]; then
    printf "[error] No existe el adapter LoRA RAG-aware: %s\n" "$RAG_AWARE_ADAPTER" >&2
    exit 1
  fi
  printf "\n[6/7] Re-inference: LoRA entrenado y evaluado con RAG...\n"
  "$PYTHON_BIN" -m src.vlm_explain_contrastive_cases \
    --dataset "$DATASET_VARIANT" \
    --method lora_rag_aware \
    --adapter "$RAG_AWARE_ADAPTER" \
    --min-explanation-tokens "$MIN_EXPLANATION_TOKENS" \
    "${LIMIT_ARGS[@]}" \
    "${DRY_RUN_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/explain_lora_rag_aware.log"
else
  printf "\n[6/7] Longest has no RAG-aware trained adapter; skipping fifth condition.\n"
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf "\n[7/7] Dry-run complete; no explanation files were generated.\n"
else
  printf "\n[7/7] Building review sheet and run summary...\n"
  "$PYTHON_BIN" -m src.summarize_contrastive_explanations \
    --dataset "$DATASET_VARIANT" \
    2>&1 | tee "$LOG_DIR/summarize.log"
fi

printf "\nDone. Outputs: %s\n" "$OUTPUT_DIR"
