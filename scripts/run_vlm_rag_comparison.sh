#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
TOP_K="${TOP_K:-3}"
RAG_RETRIEVER="${RAG_RETRIEVER:-e5_small}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
ENRICHED_ADAPTER="${ENRICHED_ADAPTER:-outputs/results/dataset_enriched/vlm_lora/final_adapter}"
LONGEST_ADAPTER="${LONGEST_ADAPTER:-outputs/results/dataset_longest_answer/vlm_lora_by_image/final_adapter}"
COMPARISON_LOG_DIR="${COMPARISON_LOG_DIR:-outputs/results/vlm_rag_comparison}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$COMPARISON_LOG_DIR"

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

run_if_needed() {
  local output_path="$1"
  shift
  if [[ "$SKIP_EXISTING" == "1" && -s "$output_path" ]]; then
    printf "\n[skip] %s ya existe\n" "$output_path"
    return 0
  fi

  printf "\n[run] %s\n" "$*"
  "$@"
}

add_if_exists() {
  local -n target_array="$1"
  local path="$2"
  if [[ -s "$path" ]]; then
    target_array+=("$path")
  fi
}

printf "\n[0/4] Ensuring by-image longest dataset exists...\n"
"$PYTHON_BIN" -m src.build_longest_by_image_dataset

for split in valid test; do
  printf "\n[1/4] Base VLM zero-shot (%s)...\n" "$split"
  run_if_needed \
    "outputs/results/dataset_enriched/vlm_zero_shot/predictions_${split}.csv" \
    "$PYTHON_BIN" -m src.vlm_infer_enriched --split "$split" "${LIMIT_ARGS[@]}"

  run_if_needed \
    "outputs/results/dataset_longest_answer/vlm_zero_shot_by_image/predictions_${split}.csv" \
    "$PYTHON_BIN" -m src.vlm_infer_longest_by_image --split "$split" "${LIMIT_ARGS[@]}"

  printf "\n[2/4] Base VLM + train-only RAG (%s)...\n" "$split"
  run_if_needed \
    "outputs/results/dataset_enriched/vlm_zero_shot_rag_${RAG_RETRIEVER}_enriched/predictions_${split}.csv" \
    "$PYTHON_BIN" -m src.vlm_rag_infer \
      --query-dataset enriched \
      --context-dataset enriched \
      --split "$split" \
      --retriever "$RAG_RETRIEVER" \
      --top-k "$TOP_K" \
      "${LIMIT_ARGS[@]}"

  run_if_needed \
    "outputs/results/dataset_longest_answer/vlm_zero_shot_by_image_rag_${RAG_RETRIEVER}_longest/predictions_${split}.csv" \
    "$PYTHON_BIN" -m src.vlm_rag_infer \
      --query-dataset longest \
      --context-dataset longest \
      --split "$split" \
      --retriever "$RAG_RETRIEVER" \
      --top-k "$TOP_K" \
      "${LIMIT_ARGS[@]}"

  printf "\n[3/4] Fine-tuned VLM + train-only RAG (%s)...\n" "$split"
  if [[ -d "$ENRICHED_ADAPTER" ]]; then
    run_if_needed \
      "outputs/results/dataset_enriched/vlm_lora_rag_${RAG_RETRIEVER}_enriched/predictions_${split}.csv" \
      "$PYTHON_BIN" -m src.vlm_rag_infer \
        --query-dataset enriched \
        --context-dataset enriched \
        --split "$split" \
        --adapter "$ENRICHED_ADAPTER" \
        --retriever "$RAG_RETRIEVER" \
        --top-k "$TOP_K" \
        "${LIMIT_ARGS[@]}"
  else
    printf "[warn] No existe adapter enriched: %s\n" "$ENRICHED_ADAPTER"
  fi

  if [[ -d "$LONGEST_ADAPTER" ]]; then
    run_if_needed \
      "outputs/results/dataset_longest_answer/vlm_lora_by_image_rag_${RAG_RETRIEVER}_longest/predictions_${split}.csv" \
      "$PYTHON_BIN" -m src.vlm_rag_infer \
        --query-dataset longest \
        --context-dataset longest \
        --split "$split" \
        --adapter "$LONGEST_ADAPTER" \
        --retriever "$RAG_RETRIEVER" \
        --top-k "$TOP_K" \
        "${LIMIT_ARGS[@]}"
  else
    printf "[warn] No existe adapter longest by-image: %s\n" "$LONGEST_ADAPTER"
  fi
done

printf "\n[4/4] Evaluating generated predictions...\n"
PREDICTIONS=()
for split in valid test; do
  add_if_exists PREDICTIONS "outputs/results/dataset_enriched/vlm_zero_shot/predictions_${split}.csv"
  add_if_exists PREDICTIONS "outputs/results/dataset_enriched/vlm_zero_shot_rag_${RAG_RETRIEVER}_enriched/predictions_${split}.csv"
  add_if_exists PREDICTIONS "outputs/results/dataset_enriched/vlm_lora_rag_${RAG_RETRIEVER}_enriched/predictions_${split}.csv"
  add_if_exists PREDICTIONS "outputs/results/dataset_longest_answer/vlm_zero_shot_by_image/predictions_${split}.csv"
  add_if_exists PREDICTIONS "outputs/results/dataset_longest_answer/vlm_zero_shot_by_image_rag_${RAG_RETRIEVER}_longest/predictions_${split}.csv"
  add_if_exists PREDICTIONS "outputs/results/dataset_longest_answer/vlm_lora_by_image_rag_${RAG_RETRIEVER}_longest/predictions_${split}.csv"
done

if [[ "${#PREDICTIONS[@]}" -gt 0 ]]; then
  "$PYTHON_BIN" -m src.evaluate_predictions "${PREDICTIONS[@]}"
else
  printf "[warn] No hay predicciones para evaluar.\n"
fi

printf "\nDone. Main outputs:\n"
printf "  Enriched metrics: outputs/metrics/dataset_enriched/metrics_mixed.csv\n"
printf "  Longest metrics: outputs/metrics/dataset_longest_answer/metrics_mixed.csv\n"
printf "  Comparison log dir: %s\n" "$COMPARISON_LOG_DIR"
