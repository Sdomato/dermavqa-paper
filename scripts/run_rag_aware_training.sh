#!/usr/bin/env bash
# RAG-aware LoRA training — enriched y longest_answer_by_image
#
# Paso 1 (CPU, sin GPU): pre-computa contextos RAG para ambos datasets
# Paso 2 (GPU):          entrena LoRA RAG-aware para cada dataset
#
# Variables de entorno:
#   PYTHON_BIN        python a usar (default: python3)
#   SKIP_BUILD        si es "1", omite el paso de build (ya está el JSONL)
#   DATASET           "enriched" | "longest" | "both" (default: both)
#   TOP_K             contextos RAG por fila (default: 3)
#   EPOCHS            épocas de entrenamiento (default: 1)
#   EVAL_STEPS        pasos entre evaluaciones (default: 50)
#   SAVE_TOTAL_LIMIT  checkpoints a conservar; 0 = todos (default: 0)
#
# Uso típico:
#   # Solo build (CPU, sin GPU — corre en máquina local)
#   SKIP_BUILD=0 DATASET=both bash scripts/run_rag_aware_training.sh --build-only
#
#   # Solo entrenar (GPU requerida — corre en VM)
#   SKIP_BUILD=1 DATASET=enriched bash scripts/run_rag_aware_training.sh
#   SKIP_BUILD=1 DATASET=longest  bash scripts/run_rag_aware_training.sh
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SKIP_BUILD="${SKIP_BUILD:-0}"
DATASET="${DATASET:-both}"
TOP_K="${TOP_K:-3}"
EPOCHS="${EPOCHS:-1}"
EVAL_STEPS="${EVAL_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-0}"

BUILD_ONLY=0
for arg in "$@"; do
  [[ "$arg" == "--build-only" ]] && BUILD_ONLY=1
done

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_build() {
  local dataset="$1"
  local out_name
  if [[ "$dataset" == "enriched" ]]; then
    out_name="outputs/datasets/dermavqa_iiyi_llm_synthesized_answer_finetune_rag_e5_small.jsonl"
  else
    out_name="outputs/datasets/dataset_longest_answer_by_image_rag_e5_small.jsonl"
  fi

  if [[ "$SKIP_BUILD" == "1" && -s "$out_name" ]]; then
    printf "[skip] Dataset RAG ya existe: %s\n" "$out_name"
    return 0
  fi

  printf "\n[build] Dataset RAG-aware para '%s' (top-k=%s)...\n" "$dataset" "$TOP_K"
  "$PYTHON_BIN" -m src.build_rag_training_dataset \
    --dataset "$dataset" \
    --top-k "$TOP_K" \
    --out "$out_name"
}

run_train() {
  local module="$1"
  local label="$2"
  printf "\n[train] %s — epochs=%s eval_steps=%s\n" "$label" "$EPOCHS" "$EVAL_STEPS"
  "$PYTHON_BIN" -m "$module" \
    --epochs "$EPOCHS" \
    --eval-steps "$EVAL_STEPS" \
    --save-total-limit "$SAVE_TOTAL_LIMIT"
}

# ── Paso 1: build datasets RAG ───────────────────────────────────────────────

if [[ "$DATASET" == "enriched" || "$DATASET" == "both" ]]; then
  run_build enriched
fi

if [[ "$DATASET" == "longest" || "$DATASET" == "both" ]]; then
  # Asegurar que el dataset by-image base exista
  if [[ ! -s "outputs/datasets/dataset_longest_answer_by_image.jsonl" ]]; then
    printf "\n[build] Generando dataset_longest_answer_by_image...\n"
    "$PYTHON_BIN" -m src.build_longest_by_image_dataset
  fi
  run_build longest
fi

[[ "$BUILD_ONLY" == "1" ]] && { printf "\n[done] Build completado. Listo para correr en GPU.\n"; exit 0; }

# ── Paso 2: entrenamiento LoRA ───────────────────────────────────────────────

if [[ "$DATASET" == "enriched" || "$DATASET" == "both" ]]; then
  run_train src.train_enriched_rag "LoRA RAG-aware — dataset_enriched"
fi

if [[ "$DATASET" == "longest" || "$DATASET" == "both" ]]; then
  run_train src.train_longest_by_image_rag "LoRA RAG-aware — dataset_longest_answer"
fi

printf "\n[done] Entrenamiento completado.\n"
printf "Adapters en:\n"
printf "  outputs/results/dataset_enriched/vlm_lora_rag_aware/final_adapter\n"
printf "  outputs/results/dataset_longest_answer/vlm_lora_by_image_rag_aware/final_adapter\n"
