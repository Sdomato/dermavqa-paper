#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_VARIANT="${DATASET_VARIANT:-enriched}"
MIN_EXPLANATION_TOKENS="${MIN_EXPLANATION_TOKENS:-100}"

case "$DATASET_VARIANT" in
  enriched)
    DATASET_NAME="dataset_enriched"
    METHODS=(zero_shot zero_shot_rag lora lora_rag lora_rag_aware)
    ;;
  longest)
    DATASET_NAME="dataset_longest_answer"
    METHODS=(zero_shot zero_shot_rag lora lora_rag)
    ;;
  *)
    printf "[error] DATASET_VARIANT debe ser enriched o longest\n" >&2
    exit 1
    ;;
esac

OUTPUT_DIR="outputs/error_analysis/$DATASET_NAME"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

for method in "${METHODS[@]}"; do
  printf "\n[external] Dataset=%s method=%s\n" "$DATASET_VARIANT" "$method"
  "$PYTHON_BIN" -m src.vlm_explain_contrastive_cases \
    --dataset "$DATASET_VARIANT" \
    --method "$method" \
    --explanation-mode external \
    --min-explanation-tokens "$MIN_EXPLANATION_TOKENS" \
    2>&1 | tee "$LOG_DIR/external_${method}.log"
done

"$PYTHON_BIN" -m src.summarize_contrastive_explanations \
  --dataset "$DATASET_VARIANT" \
  --input-dir "$OUTPUT_DIR/external_rationales" \
  --output-prefix external_rationale \
  2>&1 | tee "$LOG_DIR/summarize_external.log"

printf "\nExternal rationales complete: %s\n" "$OUTPUT_DIR"
