"""
Inferencia VLM sobre dataset_longest_answer (zero-shot y LoRA/QLoRA).

Dado un split del dataset, para cada caso arma un prompt multimodal
(imagen(es) + pregunta en español) y genera la respuesta larga con
Qwen2.5-VL. Sirve tanto para el baseline zero-shot como para inferencia
con un adapter LoRA fine-tuneado (ver --adapter).

Diseño:
  - La carga del dataset, el armado del prompt y la resolución de imágenes
    NO dependen de torch/transformers y se pueden validar en CPU con --dry-run.
  - Solo la generación real necesita GPU (Qwen2.5-VL-7B en 4-bit entra en T4/L4).

Salida (esquema canónico del plan de equipo):
  outputs/results/dataset_longest_answer/<method>/predictions_<split>.csv
  con columnas:
    split, encounter_id, image_id, question_es, reference_answer_es,
    predicted_answer_es, model_name, dataset_variant, method
  Además se guarda runtime_<split>.json con métricas operativas
  (tiempo medio por ejemplo, tiempo total, device).

Nota sobre rutas de imágenes:
  Las imágenes viven en data/iiyi/images_final/{images_train,images_valid,images_test}.
  Este script resuelve cada image_id con un rglob sobre IMAGES_DIR (abajo).
  Los baselines de retrieval resuelven igual vía retrieval_utils.resolve_image_path,
  que indexa esa misma carpeta (ver survey/STRUCTURE.md).

Uso:
    # Validación en CPU (no carga el modelo): chequea prompts e imágenes
    python -m src.vlm_infer --split valid --limit 5 --dry-run

    # Zero-shot real (requiere GPU)
    python -m src.vlm_infer --split valid
    python -m src.vlm_infer --split test

    # Inferencia con adapter LoRA fine-tuneado (Fase 3)
    python -m src.vlm_infer --split test --adapter outputs/results/dataset_longest_answer/vlm_lora/final_adapter
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.retrieval_utils import PROJECT_ROOT, build_query_text, clean_text, load_dataset

# ── paths y constantes ──────────────────────────────────────────────────────────

DATASET_PATH = PROJECT_ROOT / "outputs" / "datasets" / "dataset_longest_answer.json"
IMAGES_DIR = PROJECT_ROOT / "data" / "iiyi" / "images_final"
RESULTS_ROOT = PROJECT_ROOT / "outputs" / "results" / "dataset_longest_answer"

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DATASET_VARIANT = "longest_answer"

# Mapea el nombre de split "amigable" al valor real en el campo _split del dataset.
SPLIT_ALIASES = {
    "train": "train",
    "valid": "valid_ht",
    "test": "test_ht_spanishtestsetcorrected",
}

SYSTEM_PROMPT = (
    "Eres un dermatólogo experto. Analiza la imagen clínica del paciente "
    "y responde la consulta de forma clara y profesional."
)


# ── selección de split e items de inferencia ────────────────────────────────────

def filter_split(records: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    target = SPLIT_ALIASES.get(split, split)
    return [r for r in records if r.get("_split") == target]


def resolve_image_paths(image_ids: list[str]) -> tuple[list[str], list[str]]:
    """Resuelve cada image_id a una ruta real bajo IMAGES_DIR.

    Retorna (rutas_encontradas, ids_faltantes).
    """
    found: list[str] = []
    missing: list[str] = []
    for img_id in image_ids:
        matches = list(IMAGES_DIR.rglob(img_id)) if IMAGES_DIR.exists() else []
        if matches:
            found.append(str(matches[0]))
        else:
            missing.append(img_id)
    return found, missing


def build_inference_items(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convierte casos del dataset en items listos para inferencia.

    Cada item incluye el texto de la pregunta, la referencia, las rutas de
    imagen resueltas y los ids de imagen faltantes (para diagnóstico).
    """
    items: list[dict[str, Any]] = []
    for r in records:
        image_ids = r.get("image_ids", []) or []
        image_paths, missing = resolve_image_paths(image_ids)
        items.append(
            {
                "encounter_id": r["encounter_id"],
                "split": r.get("_split", ""),
                "image_id": ";".join(image_ids),  # esquema canónico: una columna
                "image_paths": image_paths,
                "missing_image_ids": missing,
                "question_es": build_query_text(r),
                "reference_answer_es": clean_text(r.get("answer_es", "")),
            }
        )
    return items


def build_chat_messages(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Arma el mensaje en formato chat de Qwen2.5-VL (imágenes + texto)."""
    user_content: list[dict[str, Any]] = [
        {"type": "image", "image": p} for p in item["image_paths"]
    ]
    user_content.append({"type": "text", "text": item["question_es"]})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]


# ── carga del modelo (perezosa, solo cuando se genera de verdad) ──────────────────

def load_model_and_processor(model_id: str, quantize: str | None, adapter: str | None):
    """Importa torch/transformers acá adentro para no exigirlos en --dry-run."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    bnb_config = None
    if quantize == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantize == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if bnb_config is None else None,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)

    model.eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


def generate_answer(model, processor, messages: list[dict[str, Any]], max_new_tokens: int) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # Recortar el prompt: quedarse solo con los tokens generados
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    decoded = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return decoded[0].strip()


# ── pipeline principal ───────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    records = filter_split(load_dataset(DATASET_PATH), args.split)
    if args.limit:
        records = records[: args.limit]

    items = build_inference_items(records)
    method = "vlm_lora" if args.adapter else "vlm_zero_shot"
    model_name = f"{Path(args.adapter).name}" if args.adapter else args.model

    # Diagnóstico de imágenes
    n_missing = sum(1 for it in items if it["missing_image_ids"])
    total_missing = sum(len(it["missing_image_ids"]) for it in items)
    print(f"Split '{args.split}': {len(items)} casos")
    print(f"  Imágenes faltantes: {total_missing} (en {n_missing} casos)")
    print(f"  Método: {method}  |  Modelo: {model_name}")

    if args.dry_run:
        print("\n[DRY-RUN] No se carga el modelo. Ejemplo de prompt del primer caso:")
        if items:
            example = build_chat_messages(items[0])
            print(json.dumps(example, ensure_ascii=False, indent=2)[:1200])
            print(f"\n  image_paths resueltas (caso 0): {items[0]['image_paths']}")
        return

    model, processor = load_model_and_processor(args.model, args.quantize, args.adapter)

    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    for i, item in enumerate(items, 1):
        messages = build_chat_messages(item)
        t0 = time.perf_counter()
        prediction = generate_answer(model, processor, messages, args.max_new_tokens)
        latencies.append(time.perf_counter() - t0)
        rows.append(
            {
                "split": item["split"],
                "encounter_id": item["encounter_id"],
                "image_id": item["image_id"],
                "question_es": item["question_es"],
                "reference_answer_es": item["reference_answer_es"],
                "predicted_answer_es": prediction,
                "model_name": model_name,
                "dataset_variant": DATASET_VARIANT,
                "method": method,
            }
        )
        if i % 10 == 0 or i == len(items):
            print(f"  {i}/{len(items)} generados")

    # Guardar predicciones
    out_dir = RESULTS_ROOT / method
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"predictions_{args.split}.csv"
    pd.DataFrame(rows).to_csv(pred_path, index=False, encoding="utf-8")
    print(f"\nPredicciones guardadas en {pred_path}")

    # Métricas operativas
    import statistics

    runtime = {
        "method": method,
        "model_name": model_name,
        "split": args.split,
        "n": len(rows),
        "mean_latency_s": statistics.mean(latencies) if latencies else 0.0,
        "total_time_s": sum(latencies),
        "max_new_tokens": args.max_new_tokens,
        "quantize": args.quantize,
    }
    with (out_dir / f"runtime_{args.split}.json").open("w", encoding="utf-8") as f:
        json.dump(runtime, f, ensure_ascii=False, indent=2)
    print(f"Métricas operativas: {runtime['mean_latency_s']:.2f}s/ejemplo")


# ── CLI ──────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inferencia VLM sobre dataset_longest_answer")
    p.add_argument("--split", choices=list(SPLIT_ALIASES.keys()), default="valid")
    p.add_argument("--model", default=MODEL_ID, help=f"Modelo HF (default: {MODEL_ID})")
    p.add_argument("--adapter", default=None, help="Ruta a adapter LoRA (activa method=vlm_lora)")
    p.add_argument("--quantize", choices=["4bit", "8bit"], default="4bit")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--limit", type=int, default=None, help="Procesar solo los primeros N casos")
    p.add_argument("--dry-run", action="store_true",
                   help="No carga el modelo; valida dataset, prompts e imágenes (CPU)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
