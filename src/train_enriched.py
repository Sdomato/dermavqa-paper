"""
Fine-tuning LoRA/QLoRA de Qwen2.5-VL sobre dataset_enriched.

Replica el entrenamiento de src/train_longest.py para que la comparacion contra
`dataset_longest_answer` sea directa: mismo modelo base, QLoRA 4-bit, LoRA
r=16/alpha=32, batch 1, grad_accum 16, 3 epochs, scheduler cosine y validacion
por eval_loss. La diferencia es el target: respuesta enriquecida por LLM.

Entrada de entrenamiento: imagen + question_es.
Target: synthesized_answer_es si existe, si no answer_es.
Salida:
  outputs/results/dataset_enriched/vlm_lora/final_adapter/
  outputs/results/dataset_enriched/vlm_lora/checkpoints/
  outputs/results/dataset_enriched/vlm_lora/train_runtime.json
  outputs/results/dataset_enriched/vlm_lora/train_metrics.json
  outputs/results/dataset_enriched/vlm_lora/eval_metrics_valid.json
  outputs/results/dataset_enriched/vlm_lora/trainer_state.json
  outputs/results/dataset_enriched/vlm_lora/training_log_history.{json,csv}

Uso:
    python -m src.train_enriched --dry-run --limit 5
    python -m src.train_enriched
    python -m src.train_enriched --limit 20 --epochs 1

Para entrenar + inferir valid/test + calcular metricas automaticas:
    bash scripts/run_enriched_vlm_lora.sh --epochs 1
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from src.vlm_infer_enriched import (
    MODEL_ID,
    RESULTS_ROOT,
    build_chat_messages,
    build_inference_items,
    filter_split,
    load_enriched_dataset,
)

RUN_DIR = RESULTS_ROOT / "vlm_lora"
ADAPTER_DIR = RUN_DIR / "final_adapter"
CHECKPOINTS_DIR = RUN_DIR / "checkpoints"
RUNTIME_PATH = RUN_DIR / "train_runtime.json"
TRAINING_CONFIG_PATH = RUN_DIR / "training_config.json"
TRAIN_METRICS_PATH = RUN_DIR / "train_metrics.json"
VALID_METRICS_PATH = RUN_DIR / "eval_metrics_valid.json"
TRAINER_STATE_PATH = RUN_DIR / "trainer_state.json"
LOG_HISTORY_JSON_PATH = RUN_DIR / "training_log_history.json"
LOG_HISTORY_CSV_PATH = RUN_DIR / "training_log_history.csv"

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_log_history(log_history: list[dict[str, Any]]) -> None:
    write_json(LOG_HISTORY_JSON_PATH, log_history)
    keys = sorted({key for row in log_history for key in row})
    with LOG_HISTORY_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(log_history)


def build_training_messages(item: dict[str, Any]) -> list[dict[str, Any]]:
    messages = build_chat_messages(item)
    messages.append(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": item["reference_answer_es"]}],
        }
    )
    return messages


def build_training_examples(split: str, limit: int | None, dataset_path: Path | None) -> list[dict[str, Any]]:
    records = filter_split(load_enriched_dataset(dataset_path), split)
    if limit:
        records = records[:limit]

    examples: list[dict[str, Any]] = []
    skipped = 0
    for item in build_inference_items(records):
        if not item["image_paths"] or not item["reference_answer_es"]:
            skipped += 1
            continue
        examples.append({"messages": build_training_messages(item)})

    print(f"Split '{split}': {len(examples)} ejemplos usables ({skipped} descartados)")
    return examples


def load_model_processor_and_lora(model_id: str, lora_r: int, lora_alpha: int):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        min_pixels=64 * 28 * 28,
        max_pixels=256 * 28 * 28,
    )
    return model, processor


def make_collate_fn(processor):
    from qwen_vl_utils import process_vision_info

    image_token_id = getattr(processor.tokenizer, "image_token_id", None)
    if image_token_id is None:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    pad_token_id = processor.tokenizer.pad_token_id

    def collate_fn(examples: list[dict[str, Any]]):
        messages = [example["messages"] for example in examples]

        full_texts = [
            processor.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
            for message in messages
        ]
        prompt_only = [
            processor.apply_chat_template(message[:-1], tokenize=False, add_generation_prompt=True)
            for message in messages
        ]
        image_inputs = [process_vision_info(message)[0] for message in messages]

        batch = processor(
            text=full_texts,
            images=image_inputs,
            return_tensors="pt",
            padding=True,
        )
        prompt_lens = [
            processor(text=prompt, return_tensors="pt")["input_ids"].shape[1]
            for prompt in prompt_only
        ]

        labels = batch["input_ids"].clone()
        labels[labels == pad_token_id] = -100
        if image_token_id is not None:
            labels[labels == image_token_id] = -100
        for index, prompt_len in enumerate(prompt_lens):
            labels[index, :prompt_len] = -100

        batch["labels"] = labels
        return batch

    return collate_fn


def run(args: argparse.Namespace) -> None:
    train_examples = build_training_examples("train", args.limit, args.dataset)
    eval_examples = build_training_examples("valid", args.limit, args.dataset)

    if args.dry_run:
        print("\n[DRY-RUN] No se carga el modelo. Ejemplo de chat de entrenamiento:")
        if train_examples:
            print(json.dumps(train_examples[0]["messages"], ensure_ascii=False, indent=2)[:1500])
        print(
            f"\nHiperparametros: epochs={args.epochs} lr={args.lr} "
            f"batch={args.batch_size} grad_accum={args.grad_accum} "
            f"LoRA r={args.lora_r} alpha={args.lora_alpha}"
        )
        print(f"Adapter se guardaria en: {ADAPTER_DIR}")
        return

    import torch
    from trl import SFTConfig, SFTTrainer

    model, processor = load_model_processor_and_lora(args.model, args.lora_r, args.lora_alpha)
    collate_fn = make_collate_fn(processor)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    training_config = {
        "model_name": args.model,
        "dataset_variant": "dataset_enriched",
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "eval_steps": args.eval_steps,
        "save_steps": args.eval_steps,
        "save_total_limit": args.save_total_limit,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": 0.05,
        "max_pixels": 256 * 28 * 28,
        "min_pixels": 64 * 28 * 28,
        "n_train": len(train_examples),
        "n_eval": len(eval_examples),
        "adapter_dir": str(ADAPTER_DIR),
        "checkpoints_dir": str(CHECKPOINTS_DIR),
    }
    write_json(TRAINING_CONFIG_PATH, training_config)

    sft_config = SFTConfig(
        output_dir=str(CHECKPOINTS_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=None if args.save_total_limit <= 0 else args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_examples,
        eval_dataset=eval_examples,
        data_collator=collate_fn,
        processing_class=processor.tokenizer,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    train_result = trainer.train()
    total_time = time.perf_counter() - start

    train_metrics = dict(train_result.metrics)
    train_metrics["total_train_time_s"] = round(total_time, 1)
    train_metrics["n_train"] = len(train_examples)
    train_metrics["n_eval"] = len(eval_examples)
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    write_json(TRAIN_METRICS_PATH, train_metrics)

    valid_metrics = trainer.evaluate()
    trainer.log_metrics("valid", valid_metrics)
    trainer.save_metrics("valid", valid_metrics)
    write_json(VALID_METRICS_PATH, valid_metrics)

    trainer.save_state()
    trainer.state.save_to_json(str(TRAINER_STATE_PATH))
    write_log_history(trainer.state.log_history)

    trainer.model.save_pretrained(str(ADAPTER_DIR))
    processor.save_pretrained(str(ADAPTER_DIR))
    print(f"\nAdapter LoRA guardado en {ADAPTER_DIR}")

    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    adapter_mb = sum(file.stat().st_size for file in ADAPTER_DIR.rglob("*") if file.is_file()) / 1e6
    runtime = {
        "model_name": args.model,
        "dataset_variant": "dataset_enriched",
        "n_train": len(train_examples),
        "n_eval": len(eval_examples),
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "eval_steps": args.eval_steps,
        "save_total_limit": args.save_total_limit,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "global_step": trainer.state.global_step,
        "total_train_time_s": round(total_time, 1),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "adapter_size_mb": round(adapter_mb, 1),
        "final_adapter_dir": str(ADAPTER_DIR),
        "checkpoints_dir": str(CHECKPOINTS_DIR),
        "train_metrics_path": str(TRAIN_METRICS_PATH),
        "valid_metrics_path": str(VALID_METRICS_PATH),
        "trainer_state_path": str(TRAINER_STATE_PATH),
        "log_history_json_path": str(LOG_HISTORY_JSON_PATH),
        "log_history_csv_path": str(LOG_HISTORY_CSV_PATH),
    }
    write_json(RUNTIME_PATH, runtime)
    print(
        f"Operativas -> tiempo {total_time / 60:.1f} min | "
        f"VRAM pico {peak_vram_gb:.1f} GB | adapter {adapter_mb:.0f} MB"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning LoRA/QLoRA de Qwen2.5-VL sobre dataset_enriched")
    parser.add_argument("--dataset", type=Path, default=None, help="JSONL/CSV/ZIP enriched opcional")
    parser.add_argument("--model", default=MODEL_ID, help=f"Modelo HF (default: {MODEL_ID})")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=0,
        help="Cantidad maxima de checkpoints a conservar; 0 conserva todos.",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="Usar solo las primeras N filas por split")
    parser.add_argument("--dry-run", action="store_true", help="Valida formato chat e imagenes en CPU")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
