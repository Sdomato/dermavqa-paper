"""
Shared utilities for by-image VLM datasets and inference.

Both `dataset_enriched` and `dataset_longest_answer_by_image` expose the same
row-level interface:

  split, encounter_id, image_id, image_path, question_es, answer_es

This module keeps image resolution, dataset loading, prompt construction and
generation identical across both experiments.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import time
import zipfile
from dataclasses import dataclass
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
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "data" / "iiyi" / "images_final"
LEGACY_IMAGE_ROOT = PROJECT_ROOT / "data" / "images"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
SYSTEM_PROMPT = (
    "Eres un dermatologo experto. Analiza la imagen clinica del paciente "
    "y responde la consulta en espanol de forma clara, concisa y prudente."
)


@dataclass(frozen=True)
class ByImageDatasetConfig:
    dataset_name: str
    dataset_variant: str
    default_paths: tuple[Path, ...]
    results_root: Path
    answer_columns: tuple[str, ...]
    lora_method: str
    zero_shot_method: str
    split_aliases: dict[str, str]
    missing_message: str


class ImageResolver:
    def __init__(self, project_root: Path = PROJECT_ROOT) -> None:
        env_root = Path(os.environ.get("DERMAVQA_IMAGE_ROOT", "")).expanduser()
        self.image_roots = (
            env_root if str(env_root) else DEFAULT_IMAGE_ROOT,
            LEGACY_IMAGE_ROOT,
        )
        self.project_root = project_root
        self._index: dict[str, str] | None = None

    def _build_index(self) -> dict[str, str]:
        index: dict[str, str] = {}
        for base in self.image_roots:
            if not base.exists():
                continue
            for image_path in base.rglob("*"):
                if image_path.is_file():
                    index.setdefault(image_path.name, str(image_path.resolve()))
        return index

    @property
    def index(self) -> dict[str, str]:
        if self._index is None:
            self._index = self._build_index()
        return self._index

    def resolve(self, record: dict[str, Any]) -> tuple[str | None, str | None]:
        image_id = clean_text(record.get("image_id", ""))
        if image_id and image_id in self.index:
            return self.index[image_id], None

        raw_path = clean_text(record.get("image_path", ""))
        if raw_path:
            path = Path(raw_path)
            if not path.is_absolute():
                path = self.project_root / path
            if path.exists():
                return str(path.resolve()), None
            return None, image_id or raw_path

        return None, image_id or None


def select_default_path(paths: tuple[Path, ...]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[-1]


def load_by_image_dataset(
    path: Path | None,
    default_paths: tuple[Path, ...],
    missing_message: str,
) -> list[dict[str, Any]]:
    candidate = path or select_default_path(default_paths)
    if not candidate.exists():
        raise FileNotFoundError(missing_message)

    suffix = candidate.suffix.lower()
    if suffix == ".jsonl":
        with candidate.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    if suffix == ".json":
        with candidate.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    if suffix == ".csv":
        import pandas as pd

        return pd.read_csv(candidate).fillna("").to_dict(orient="records")

    if suffix == ".zip":
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
            json_name = next((name for name in names if name.endswith(".json")), None)
            if json_name:
                with archive.open(json_name) as handle:
                    return json.loads(handle.read().decode("utf-8"))
            csv_name = next((name for name in names if name.endswith(".csv")), None)
            if csv_name:
                with archive.open(csv_name) as handle:
                    reader = csv.DictReader(line.decode("utf-8") for line in handle)
                    return [dict(row) for row in reader]
        raise ValueError(f"El zip {candidate} no contiene JSON/JSONL/CSV usable")

    raise ValueError(f"Formato de dataset no soportado: {candidate}")


def filter_split(
    records: list[dict[str, Any]],
    split: str,
    split_aliases: dict[str, str],
    dataset_name: str,
) -> list[dict[str, Any]]:
    target = split_aliases.get(split, split)
    result = [record for record in records if str(record.get("split", "")) == target]
    if not result:
        raise ValueError(f"Split '{split}' no encontrado en {dataset_name}")
    return result


def reference_answer(record: dict[str, Any], answer_columns: tuple[str, ...]) -> str:
    for column in answer_columns:
        value = clean_text(record.get(column, ""))
        if value:
            return value
    return ""


def build_inference_items(
    records: list[dict[str, Any]],
    answer_columns: tuple[str, ...],
    resolver: ImageResolver | None = None,
) -> list[dict[str, Any]]:
    resolver = resolver or ImageResolver()
    items: list[dict[str, Any]] = []
    for record in records:
        image_path, missing = resolver.resolve(record)
        items.append(
            {
                "encounter_id": clean_text(record.get("encounter_id", "")),
                "split": record.get("split", ""),
                "image_id": clean_text(record.get("image_id", "")),
                "image_path": image_path or "",
                "image_paths": [image_path] if image_path else [],
                "missing_image_ids": [missing] if missing else [],
                "question_es": clean_text(record.get("question_es", "")),
                "reference_answer_es": reference_answer(record, answer_columns),
            }
        )
    return items


def build_chat_messages(
    item: dict[str, Any],
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = [
        {"type": "image", "image": image_path} for image_path in item["image_paths"]
    ]
    user_content.append({"type": "text", "text": item["question_es"]})
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
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


RAG_SYSTEM_PROMPT = (
    "Eres un dermatologo experto. Analiza la imagen clinica del paciente "
    "y responde la consulta en espanol de forma clara, concisa y prudente. "
    "Usa los casos similares recuperados solo como contexto auxiliar. "
    "No los menciones como fuente, no copies respuestas si no corresponden "
    "y prioriza la imagen y la consulta actual."
)


def format_rag_prompt(question: str, contexts: list[dict[str, Any]]) -> str:
    lines = [
        "Consulta actual:",
        clean_text(question),
        "",
        "Casos similares recuperados del conjunto de entrenamiento:",
    ]
    for i, ctx in enumerate(contexts, start=1):
        lines.extend([
            f"{i}. Pregunta similar: {ctx['question_es']}",
            f"   Respuesta del caso similar: {ctx['answer_es']}",
        ])
    lines.extend([
        "",
        "Responde ahora la consulta actual en espanol clinico, claro y prudente.",
    ])
    return "\n".join(lines)


def build_rag_inference_items(
    records: list[dict[str, Any]],
    answer_columns: tuple[str, ...],
    resolver: ImageResolver | None = None,
) -> list[dict[str, Any]]:
    """Como build_inference_items pero propaga el campo rag_contexts del registro."""
    items = build_inference_items(records, answer_columns, resolver)
    for item, record in zip(items, records):
        raw = record.get("rag_contexts")
        if isinstance(raw, str):
            try:
                item["rag_contexts"] = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                item["rag_contexts"] = []
        elif isinstance(raw, list):
            item["rag_contexts"] = raw
        else:
            item["rag_contexts"] = []
    return items


def build_rag_chat_messages(
    item: dict[str, Any],
    system_prompt: str = RAG_SYSTEM_PROMPT,
) -> list[dict[str, Any]]:
    """Prompt con contexto RAG pre-computado en item['rag_contexts']."""
    contexts = item.get("rag_contexts") or []
    user_content: list[dict[str, Any]] = [
        {"type": "image", "image": image_path} for image_path in item["image_paths"]
    ]
    user_content.append({"type": "text", "text": format_rag_prompt(item["question_es"], contexts)})
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]


def run_by_image_inference(args: Any, config: ByImageDatasetConfig) -> None:
    records = filter_split(
        load_by_image_dataset(args.dataset, config.default_paths, config.missing_message),
        args.split,
        config.split_aliases,
        config.dataset_name,
    )
    if args.limit:
        records = records[: args.limit]

    items = build_inference_items(records, config.answer_columns)
    method = config.lora_method if args.adapter else config.zero_shot_method
    model_name = Path(args.adapter).name if args.adapter else args.model

    n_missing = sum(1 for item in items if item["missing_image_ids"])
    total_missing = sum(len(item["missing_image_ids"]) for item in items)
    print(f"Split '{args.split}': {len(items)} filas por imagen")
    print(f"  Imagenes faltantes: {total_missing} (en {n_missing} filas)")
    print(f"  Metodo: {method} | Modelo: {model_name}")

    if args.dry_run:
        print("\n[DRY-RUN] No se carga el modelo. Ejemplo de prompt:")
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
                "dataset_variant": config.dataset_variant,
                "method": method,
            }
        )
        if index % 10 == 0 or index == len(items):
            print(f"  {index}/{len(items)} generados")

    out_dir = config.results_root / method
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"predictions_{args.split}.csv"

    import pandas as pd

    pd.DataFrame(rows).to_csv(pred_path, index=False, encoding="utf-8")
    print(f"\nPredicciones guardadas en {pred_path}")

    runtime = {
        "method": method,
        "model_name": model_name,
        "dataset_variant": config.dataset_variant,
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
