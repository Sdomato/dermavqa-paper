"""
Fine-tuning LoRA/QLoRA de Qwen2.5-VL sobre dataset_longest_answer (Fase 2).

Entrena el mismo VLM que usa src/vlm_infer.py, pero fine-tuneado sobre la
respuesta larga (answer_es) del split `train`, usando `valid_ht` para elegir
checkpoint. Guarda SOLO el adapter LoRA (no el modelo base) para que la
inferencia (Fase 3) lo cargue con `vlm_infer --adapter <ruta>`.

Diseño (consistente con vlm_infer.py):
  - La carga del dataset, el filtrado por split y el armado del formato chat
    NO dependen de torch/transformers y se validan en CPU con --dry-run.
  - Solo el entrenamiento real necesita GPU. QLoRA 4-bit entra en una T4/L4;
    para márgenes cómodos conviene L4/A10 (24 GB).

Reusa de src/vlm_infer.py todo lo que ya está validado:
  MODEL_ID, SPLIT_ALIASES, SYSTEM_PROMPT, DATASET_PATH, RESULTS_ROOT,
  filter_split(), build_inference_items() y build_chat_messages().

Salida:
  outputs/results/dataset_longest_answer/vlm_lora/final_adapter/   (adapter LoRA)
  outputs/results/dataset_longest_answer/vlm_lora/train_runtime.json (operativas:
    tiempo total de fine-tuning, VRAM pico, tamaño del adapter en MB).

Hiperparámetros de arranque (plan Fase 2):
  3 epochs · LR 2e-4 · batch 1 + grad_accum 16 · LoRA r=16 α=32 · QLoRA 4-bit ·
  load_best_model_at_end sobre valid_ht.

Uso:
    # Validación en CPU (no carga el modelo): chequea formato chat e imágenes
    python -m src.train_longest --dry-run --limit 5

    # Fine-tuning real (requiere GPU)
    python -m src.train_longest

    # Prueba rápida en GPU (pocos pasos)
    python -m src.train_longest --limit 20 --epochs 1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from src.retrieval_utils import load_dataset
from src.vlm_infer import (
    DATASET_PATH,
    MODEL_ID,
    RESULTS_ROOT,
    build_chat_messages,
    build_inference_items,
    filter_split,
)

# ── paths y constantes ──────────────────────────────────────────────────────────

ADAPTER_DIR = RESULTS_ROOT / "vlm_lora" / "final_adapter"
RUNTIME_PATH = RESULTS_ROOT / "vlm_lora" / "train_runtime.json"

# Módulos lineales típicos de Qwen2.5-VL donde se inyecta LoRA (torre de lenguaje).
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ── armado de ejemplos de entrenamiento ──────────────────────────────────────────

def build_training_messages(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Mensaje chat completo: system + user (imágenes + pregunta) + assistant (target).

    Reusa build_chat_messages() (system + user, idéntico a inferencia) y le
    agrega el turno assistant con la respuesta larga de referencia, que es
    sobre lo único que se calcula la loss.
    """
    # Limitar a 1 imagen para reducir memoria en T4 16GB
    item_1img = {**item, "image_paths": item["image_paths"][:1]}
    messages = build_chat_messages(item_1img)
    messages.append(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": item["reference_answer_es"]}],
        }
    )
    return messages


def build_training_examples(split: str, limit: int | None) -> list[dict[str, Any]]:
    """Casos de un split → ejemplos {'messages': ...} listos para el collator.

    Descarta casos sin imagen resuelta o sin respuesta de referencia (no aportan
    señal de entrenamiento y romperían el procesado de visión).
    """
    records = filter_split(load_dataset(DATASET_PATH), split)
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


# ── modelo y collator (perezoso, solo con GPU) ───────────────────────────────────

def load_model_processor_and_lora(model_id: str, lora_r: int, lora_alpha: int):
    """QLoRA 4-bit + LoraConfig. Importa torch/peft/transformers acá adentro."""
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
    model.config.use_cache = False  # incompatible con gradient checkpointing

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
        max_pixels=256 * 28 * 28,  # reducido para caber en T4 16GB con 3B
    )
    return model, processor


def make_collate_fn(processor):
    """Collator multimodal: tokeniza el chat y enmascara prompt e imágenes en la loss.

    Solo los tokens del turno assistant (la respuesta de referencia) contribuyen
    a la loss. El prompt completo (system + user + placeholders de imagen + padding)
    se enmascara con -100.
    """
    from qwen_vl_utils import process_vision_info

    image_token_id = getattr(processor.tokenizer, "image_token_id", None)
    if image_token_id is None:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    pad_token_id = processor.tokenizer.pad_token_id

    def collate_fn(examples: list[dict[str, Any]]):
        messages = [ex["messages"] for ex in examples]

        # Texto completo (system + user + assistant) para el forward pass.
        full_texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in messages
        ]
        # Texto solo del prompt (system + user) para medir su longitud en tokens
        # y enmascarar esos tokens en la loss.
        prompt_only = [
            processor.apply_chat_template(m[:-1], tokenize=False, add_generation_prompt=True)
            for m in messages
        ]

        image_inputs = [process_vision_info(m)[0] for m in messages]

        batch = processor(
            text=full_texts,
            images=image_inputs,
            return_tensors="pt",
            padding=True,
        )
        prompt_lens = [
            processor(text=p, return_tensors="pt")["input_ids"].shape[1]
            for p in prompt_only
        ]

        labels = batch["input_ids"].clone()
        # Enmascarar padding e image tokens.
        labels[labels == pad_token_id] = -100
        if image_token_id is not None:
            labels[labels == image_token_id] = -100
        # Enmascarar el prompt (system + user) fila a fila.
        for i, prompt_len in enumerate(prompt_lens):
            labels[i, :prompt_len] = -100

        batch["labels"] = labels
        return batch

    return collate_fn


# ── pipeline principal ───────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    train_examples = build_training_examples("train", args.limit)
    eval_examples = build_training_examples("valid", args.limit)

    if args.dry_run:
        print("\n[DRY-RUN] No se carga el modelo. Ejemplo de chat de entrenamiento (caso 0):")
        if train_examples:
            print(
                json.dumps(train_examples[0]["messages"], ensure_ascii=False, indent=2)[:1500]
            )
        print(
            f"\nHiperparámetros: epochs={args.epochs} lr={args.lr} "
            f"batch={args.batch_size} grad_accum={args.grad_accum} "
            f"LoRA r={args.lora_r} α={args.lora_alpha}"
        )
        print(f"Adapter se guardaría en: {ADAPTER_DIR}")
        return

    import torch
    from trl import SFTConfig, SFTTrainer

    model, processor = load_model_processor_and_lora(
        args.model, args.lora_r, args.lora_alpha
    )
    collate_fn = make_collate_fn(processor)

    ADAPTER_DIR.parent.mkdir(parents=True, exist_ok=True)
    sft_config = SFTConfig(
        output_dir=str(ADAPTER_DIR.parent / "checkpoints"),
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
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,  # el collator necesita 'messages'
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
    t0 = time.perf_counter()
    trainer.train()
    total_time = time.perf_counter() - t0

    # Guardar SOLO el adapter (no el modelo base).
    trainer.model.save_pretrained(str(ADAPTER_DIR))
    processor.save_pretrained(str(ADAPTER_DIR))
    print(f"\nAdapter LoRA guardado en {ADAPTER_DIR}")

    # Métricas operativas.
    peak_vram_gb = (
        torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    )
    adapter_mb = sum(f.stat().st_size for f in ADAPTER_DIR.rglob("*") if f.is_file()) / 1e6
    runtime = {
        "model_name": args.model,
        "n_train": len(train_examples),
        "n_eval": len(eval_examples),
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "total_train_time_s": round(total_time, 1),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "adapter_size_mb": round(adapter_mb, 1),
    }
    with RUNTIME_PATH.open("w", encoding="utf-8") as f:
        json.dump(runtime, f, ensure_ascii=False, indent=2)
    print(
        f"Operativas → tiempo {total_time / 60:.1f} min · "
        f"VRAM pico {peak_vram_gb:.1f} GB · adapter {adapter_mb:.0f} MB"
    )


# ── CLI ──────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tuning LoRA/QLoRA de Qwen2.5-VL sobre dataset_longest_answer"
    )
    p.add_argument("--model", default=MODEL_ID, help=f"Modelo HF (default: {MODEL_ID})")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--limit", type=int, default=None,
                   help="Usar solo los primeros N casos por split (pruebas rápidas)")
    p.add_argument("--dry-run", action="store_true",
                   help="No carga el modelo; valida formato chat e imágenes (CPU)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
