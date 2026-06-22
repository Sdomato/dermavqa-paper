"""
Inferencia VLM sobre dataset_enriched (zero-shot y LoRA/QLoRA).

Copia el flujo de src/vlm_infer.py usado para dataset_longest_answer, pero usa
el dataset enriquecido por imagen:

  outputs/datasets/dermavqa_iiyi_llm_synthesized_answer_finetune.{jsonl,csv,zip}

Entrada: imagen + question_es.
Referencia: synthesized_answer_es si existe, si no answer_es.
Salida:
  outputs/results/dataset_enriched/<method>/predictions_<split>.csv
  outputs/results/dataset_enriched/<method>/runtime_<split>.json

Uso:
    python -m src.vlm_infer_enriched --split valid --limit 5 --dry-run
    python -m src.vlm_infer_enriched --split valid
    python -m src.vlm_infer_enriched --split test
    python -m src.vlm_infer_enriched --split test \
        --adapter outputs/results/dataset_enriched/vlm_lora/final_adapter
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Any


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path(__file__).parent).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "outputs" / "datasets").exists():
            return candidate
    return start


def clean_text(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


PROJECT_ROOT = find_project_root()

DATASETS_DIR = PROJECT_ROOT / "outputs" / "datasets"
DATASET_JSONL_PATH = DATASETS_DIR / "dermavqa_iiyi_llm_synthesized_answer_finetune.jsonl"
DATASET_CSV_PATH = DATASETS_DIR / "dermavqa_iiyi_llm_synthesized_answer_finetune.csv"
DATASET_ZIP_PATH = DATASETS_DIR / "dermavqa_iiyi_llm_synthesized_answer_finetune.zip"

IMAGES_DIR = Path(os.environ.get("DERMAVQA_IMAGE_ROOT", "")).expanduser()
if not str(IMAGES_DIR):
    IMAGES_DIR = PROJECT_ROOT / "data" / "iiyi" / "images_final"
_LEGACY_IMAGES_DIR = PROJECT_ROOT / "data" / "images"

RESULTS_ROOT = PROJECT_ROOT / "outputs" / "results" / "dataset_enriched"

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_VARIANT = "dataset_enriched"
ANSWER_COLUMNS = ("synthesized_answer_es", "answer_es")

SPLIT_ALIASES = {
    "train": "train",
    "valid": "valid",
    "test": "test",
}

SYSTEM_PROMPT = (
    "Eres un dermatologo experto. Analiza la imagen clinica del paciente "
    "y responde la consulta en espanol de forma clara, concisa y prudente."
)


def load_enriched_dataset(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the enriched dataset from JSONL/CSV, falling back to the zip."""
    candidate = path
    if candidate is None:
        if DATASET_JSONL_PATH.exists():
            candidate = DATASET_JSONL_PATH
        elif DATASET_CSV_PATH.exists():
            candidate = DATASET_CSV_PATH
        else:
            candidate = DATASET_ZIP_PATH

    if not candidate.exists():
        raise FileNotFoundError(
            "No encontre el dataset enriched. Esperaba uno de: "
            f"{DATASET_JSONL_PATH}, {DATASET_CSV_PATH}, {DATASET_ZIP_PATH}"
        )

    if candidate.suffix.lower() == ".jsonl":
        with candidate.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    if candidate.suffix.lower() == ".csv":
        import pandas as pd

        return pd.read_csv(candidate).fillna("").to_dict(orient="records")

    if candidate.suffix.lower() == ".zip":
        with zipfile.ZipFile(candidate) as archive:
            names = archive.namelist()
            jsonl_name = next((name for name in names if name.endswith(".jsonl")), None)
            if jsonl_name:
                with archive.open(jsonl_name) as handle:
                    return [
                        json.loads(line.decode("utf-8"))
                        for line in handle
                        if line.strip()
                    ]
            csv_name = next((name for name in names if name.endswith(".csv")), None)
            if csv_name:
                with archive.open(csv_name) as handle:
                    reader = csv.DictReader(
                        line.decode("utf-8") for line in handle
                    )
                    return [dict(row) for row in reader]
        raise ValueError(f"El zip {candidate} no contiene JSONL ni CSV usable")

    raise ValueError(f"Formato de dataset no soportado: {candidate}")


def filter_split(records: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    target = SPLIT_ALIASES.get(split, split)
    result = [record for record in records if str(record.get("split", "")) == target]
    if not result:
        raise ValueError(f"Split '{split}' no encontrado en dataset_enriched")
    return result


def _build_image_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for base in (IMAGES_DIR, _LEGACY_IMAGES_DIR):
        if not base.exists():
            continue
        for image_path in base.rglob("*"):
            if image_path.is_file():
                index.setdefault(image_path.name, str(image_path.resolve()))
    return index


_IMAGE_INDEX: dict[str, str] = {}


def resolve_image_path(record: dict[str, Any]) -> tuple[str | None, str | None]:
    """Resolve one enriched row image using image_id, env root, or stored path."""
    global _IMAGE_INDEX
    if not _IMAGE_INDEX:
        _IMAGE_INDEX = _build_image_index()

    image_id = clean_text(record.get("image_id", ""))
    if image_id and image_id in _IMAGE_INDEX:
        return _IMAGE_INDEX[image_id], None

    raw_path = clean_text(record.get("image_path", ""))
    if raw_path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.exists():
            return str(path.resolve()), None
        if image_id:
            missing = image_id
        else:
            missing = raw_path
    else:
        missing = image_id or None

    return None, missing


def reference_answer(record: dict[str, Any]) -> str:
    for column in ANSWER_COLUMNS:
        value = clean_text(record.get(column, ""))
        if value:
            return value
    return ""


def build_inference_items(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for record in records:
        image_path, missing = resolve_image_path(record)
        items.append(
            {
                "encounter_id": record["encounter_id"],
                "split": record.get("split", ""),
                "image_id": clean_text(record.get("image_id", "")),
                "image_path": image_path or "",
                "image_paths": [image_path] if image_path else [],
                "missing_image_ids": [missing] if missing else [],
                "question_es": clean_text(record.get("question_es", "")),
                "reference_answer_es": reference_answer(record),
            }
        )
    return items


def build_chat_messages(item: dict[str, Any]) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = [
        {"type": "image", "image": image_path} for image_path in item["image_paths"]
    ]
    user_content.append({"type": "text", "text": item["question_es"]})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]


def load_model_and_processor(model_id: str, quantize: str | None, adapter: str | None):
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
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        min_pixels=64 * 28 * 28,
        max_pixels=256 * 28 * 28,
    )
    return model, processor


def generate_answer(model, processor, messages: list[dict[str, Any]], max_new_tokens: int) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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

    trimmed = [output[len(input_ids):] for input_ids, output in zip(inputs.input_ids, generated)]
    decoded = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return decoded[0].strip()


def run(args: argparse.Namespace) -> None:
    records = filter_split(load_enriched_dataset(args.dataset), args.split)
    if args.limit:
        records = records[: args.limit]

    items = build_inference_items(records)
    method = "vlm_lora" if args.adapter else "vlm_zero_shot"
    model_name = Path(args.adapter).name if args.adapter else args.model

    n_missing = sum(1 for item in items if item["missing_image_ids"])
    total_missing = sum(len(item["missing_image_ids"]) for item in items)
    print(f"Split '{args.split}': {len(items)} filas por imagen")
    print(f"  Imagenes faltantes: {total_missing} (en {n_missing} filas)")
    print(f"  Metodo: {method} | Modelo: {model_name}")

    if args.dry_run:
        print("\n[DRY-RUN] No se carga el modelo. Ejemplo de prompt del primer caso:")
        if items:
            example = build_chat_messages(items[0])
            print(json.dumps(example, ensure_ascii=False, indent=2)[:1200])
            print(f"\n  image_path resuelta: {items[0]['image_path']}")
            print(f"  reference_answer_es: {items[0]['reference_answer_es'][:300]}")
        return

    model, processor = load_model_and_processor(args.model, args.quantize, args.adapter)

    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    for index, item in enumerate(items, 1):
        messages = build_chat_messages(item)
        start = time.perf_counter()
        prediction = generate_answer(model, processor, messages, args.max_new_tokens)
        latencies.append(time.perf_counter() - start)
        rows.append(
            {
                "split": item["split"],
                "encounter_id": item["encounter_id"],
                "image_id": item["image_id"],
                "image_path": item["image_path"],
                "question_es": item["question_es"],
                "reference_answer_es": item["reference_answer_es"],
                "predicted_answer_es": prediction,
                "model_name": model_name,
                "dataset_variant": DATASET_VARIANT,
                "method": method,
            }
        )
        if index % 10 == 0 or index == len(items):
            print(f"  {index}/{len(items)} generados")

    out_dir = RESULTS_ROOT / method
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"predictions_{args.split}.csv"
    import pandas as pd

    pd.DataFrame(rows).to_csv(pred_path, index=False, encoding="utf-8")
    print(f"\nPredicciones guardadas en {pred_path}")

    import statistics

    runtime = {
        "method": method,
        "model_name": model_name,
        "dataset_variant": DATASET_VARIANT,
        "split": args.split,
        "n": len(rows),
        "mean_latency_s": statistics.mean(latencies) if latencies else 0.0,
        "total_time_s": sum(latencies),
        "max_new_tokens": args.max_new_tokens,
        "quantize": args.quantize,
    }
    with (out_dir / f"runtime_{args.split}.json").open("w", encoding="utf-8") as handle:
        json.dump(runtime, handle, ensure_ascii=False, indent=2)
    print(f"Metricas operativas: {runtime['mean_latency_s']:.2f}s/ejemplo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inferencia VLM sobre dataset_enriched")
    parser.add_argument("--split", choices=list(SPLIT_ALIASES.keys()), default="valid")
    parser.add_argument("--dataset", type=Path, default=None, help="JSONL/CSV/ZIP enriched opcional")
    parser.add_argument("--model", default=MODEL_ID, help=f"Modelo HF (default: {MODEL_ID})")
    parser.add_argument("--adapter", default=None, help="Ruta a adapter LoRA (activa method=vlm_lora)")
    parser.add_argument("--quantize", choices=["4bit", "8bit"], default="4bit")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="Procesar solo las primeras N filas")
    parser.add_argument("--dry-run", action="store_true", help="Valida dataset, prompts e imagenes en CPU")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
