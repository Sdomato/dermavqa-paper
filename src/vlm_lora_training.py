"""
Shared QLoRA training engine for Qwen2.5-VL by-image experiments.

Dataset-specific scripts only provide loaders and output locations. The model,
collator, LoRA config, optimizer schedule, metrics and runtime artifacts live
here so `dataset_enriched` and `dataset_longest_answer_by_image` stay
reproducible and comparable.
"""

from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

RecordsLoader = Callable[[Path | None], list[dict[str, Any]]]
SplitFilter = Callable[[list[dict[str, Any]], str], list[dict[str, Any]]]
ItemBuilder = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
ChatBuilder = Callable[[dict[str, Any]], list[dict[str, Any]]]


@dataclass(frozen=True)
class VlmLoraTrainConfig:
    dataset_variant: str
    comparison_protocol: str
    training_unit: str
    run_dir: Path
    adapter_dir: Path
    checkpoints_dir: Path
    runtime_path: Path
    training_config_path: Path
    train_metrics_path: Path
    valid_metrics_path: Path
    trainer_state_path: Path
    log_history_json_path: Path
    log_history_csv_path: Path
    load_records: RecordsLoader
    filter_split: SplitFilter
    build_inference_items: ItemBuilder
    build_chat_messages: ChatBuilder


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_log_history(config: VlmLoraTrainConfig, log_history: list[dict[str, Any]]) -> None:
    write_json(config.log_history_json_path, log_history)
    keys = sorted({key for row in log_history for key in row})
    with config.log_history_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(log_history)


def set_reproducibility_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def build_training_messages(
    item: dict[str, Any],
    build_chat_messages: ChatBuilder,
) -> list[dict[str, Any]]:
    messages = build_chat_messages(item)
    messages.append(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": item["reference_answer_es"]}],
        }
    )
    return messages


def build_training_examples(
    split: str,
    limit: int | None,
    dataset_path: Path | None,
    config: VlmLoraTrainConfig,
) -> list[dict[str, Any]]:
    records = config.filter_split(config.load_records(dataset_path), split)
    if limit:
        records = records[:limit]

    examples: list[dict[str, Any]] = []
    skipped = 0
    for item in config.build_inference_items(records):
        if not item["image_paths"] or not item["reference_answer_es"]:
            skipped += 1
            continue
        examples.append(
            {"messages": build_training_messages(item, config.build_chat_messages)}
        )

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


def run_lora_training(args: Any, config: VlmLoraTrainConfig) -> None:
    set_reproducibility_seed(args.seed)
    train_examples = build_training_examples("train", args.limit, args.dataset, config)
    eval_examples = build_training_examples("valid", args.limit, args.dataset, config)

    if args.dry_run:
        print("\n[DRY-RUN] No se carga el modelo. Ejemplo de chat de entrenamiento:")
        if train_examples:
            print(json.dumps(train_examples[0]["messages"], ensure_ascii=False, indent=2)[:1500])
        print(
            f"\nHiperparametros: epochs={args.epochs} lr={args.lr} "
            f"batch={args.batch_size} grad_accum={args.grad_accum} "
            f"LoRA r={args.lora_r} alpha={args.lora_alpha} seed={args.seed}"
        )
        print(f"Adapter se guardaria en: {config.adapter_dir}")
        return

    import torch
    from trl import SFTConfig, SFTTrainer

    try:
        from transformers import set_seed

        set_seed(args.seed)
    except Exception:
        pass

    model, processor = load_model_processor_and_lora(args.model, args.lora_r, args.lora_alpha)
    collate_fn = make_collate_fn(processor)

    config.run_dir.mkdir(parents=True, exist_ok=True)
    estimated_optimizer_steps = math.ceil(
        len(train_examples) * args.epochs / (args.batch_size * args.grad_accum)
    )
    training_config = {
        "model_name": args.model,
        "dataset_variant": config.dataset_variant,
        "comparison_protocol": config.comparison_protocol,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "seed": args.seed,
        "estimated_optimizer_steps": estimated_optimizer_steps,
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
        "training_unit": config.training_unit,
        "adapter_dir": str(config.adapter_dir),
        "checkpoints_dir": str(config.checkpoints_dir),
    }
    write_json(config.training_config_path, training_config)

    sft_config = SFTConfig(
        output_dir=str(config.checkpoints_dir),
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
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,
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
    write_json(config.train_metrics_path, train_metrics)

    valid_metrics = trainer.evaluate()
    trainer.log_metrics("valid", valid_metrics)
    trainer.save_metrics("valid", valid_metrics)
    write_json(config.valid_metrics_path, valid_metrics)

    trainer.save_state()
    trainer.state.save_to_json(str(config.trainer_state_path))
    write_log_history(config, trainer.state.log_history)

    trainer.model.save_pretrained(str(config.adapter_dir))
    processor.save_pretrained(str(config.adapter_dir))
    print(f"\nAdapter LoRA guardado en {config.adapter_dir}")

    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    adapter_mb = sum(file.stat().st_size for file in config.adapter_dir.rglob("*") if file.is_file()) / 1e6
    runtime = {
        "model_name": args.model,
        "dataset_variant": config.dataset_variant,
        "comparison_protocol": config.comparison_protocol,
        "n_train": len(train_examples),
        "n_eval": len(eval_examples),
        "training_unit": config.training_unit,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "seed": args.seed,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "eval_steps": args.eval_steps,
        "save_total_limit": args.save_total_limit,
        "estimated_optimizer_steps": estimated_optimizer_steps,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "global_step": trainer.state.global_step,
        "total_train_time_s": round(total_time, 1),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "adapter_size_mb": round(adapter_mb, 1),
        "final_adapter_dir": str(config.adapter_dir),
        "checkpoints_dir": str(config.checkpoints_dir),
        "train_metrics_path": str(config.train_metrics_path),
        "valid_metrics_path": str(config.valid_metrics_path),
        "trainer_state_path": str(config.trainer_state_path),
        "log_history_json_path": str(config.log_history_json_path),
        "log_history_csv_path": str(config.log_history_csv_path),
    }
    write_json(config.runtime_path, runtime)
    print(
        f"Operativas -> tiempo {total_time / 60:.1f} min | "
        f"VRAM pico {peak_vram_gb:.1f} GB | adapter {adapter_mb:.0f} MB"
    )
