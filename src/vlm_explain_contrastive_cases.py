"""Re-infer selected cases with a structured, evidence-based explanation.

The generated explanation is a post-hoc observable justification, not a
faithful trace of hidden chain-of-thought. References are deliberately kept
out of the model prompt so they cannot leak into the answer.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from src.vlm_by_image_utils import (
    MODEL_ID,
    PROJECT_ROOT,
    ImageResolver,
    clean_text,
    format_rag_prompt,
    generate_answer,
    load_model_and_processor,
)


METHODS = (
    "zero_shot",
    "zero_shot_rag",
    "lora",
    "lora_rag",
    "lora_rag_aware",
)
RAG_METHODS = {"zero_shot_rag", "lora_rag", "lora_rag_aware"}
DATASET_NAMES = {
    "enriched": "dataset_enriched",
    "longest": "dataset_longest_answer",
}
ORIGINAL_PREDICTION_PATHS = {
    ("enriched", "zero_shot"): PROJECT_ROOT
    / "outputs/results/dataset_enriched/vlm_zero_shot/predictions_test.csv",
    ("enriched", "zero_shot_rag"): PROJECT_ROOT
    / "outputs/results/dataset_enriched/"
    "vlm_zero_shot_rag_e5_small_enriched/predictions_test.csv",
    ("enriched", "lora"): PROJECT_ROOT
    / "outputs/results/dataset_enriched/vlm_lora/predictions_test.csv",
    ("enriched", "lora_rag"): PROJECT_ROOT
    / "outputs/results/dataset_enriched/"
    "vlm_lora_rag_e5_small_enriched/predictions_test.csv",
    ("enriched", "lora_rag_aware"): PROJECT_ROOT
    / "outputs/results/dataset_enriched/"
    "vlm_lora_rag_aware/predictions_test.csv",
    ("longest", "zero_shot"): PROJECT_ROOT
    / "outputs/results/dataset_longest_answer/"
    "vlm_zero_shot_by_image/predictions_test.csv",
    ("longest", "zero_shot_rag"): PROJECT_ROOT
    / "outputs/results/dataset_longest_answer/"
    "vlm_zero_shot_by_image_rag_e5_small_longest/predictions_test.csv",
    ("longest", "lora"): PROJECT_ROOT
    / "outputs/results/dataset_longest_answer/"
    "vlm_lora_by_image/predictions_test.csv",
    ("longest", "lora_rag"): PROJECT_ROOT
    / "outputs/results/dataset_longest_answer/"
    "vlm_lora_by_image_rag_e5_small_longest/predictions_test.csv",
}
SYSTEM_PROMPT = (
    "Eres un asistente dermatologico experto. Responde en espanol clinico, "
    "claro y prudente. No describas razonamiento interno paso a paso ni una "
    "cadena de pensamiento. En cambio, entrega una justificacion breve y "
    "verificable basada solo en hallazgos observables de la imagen, datos "
    "explicitos de la consulta y, cuando existan, casos recuperados. No "
    "inventes signos, antecedentes, diagnosticos ni tratamientos."
)
EXTERNAL_SYSTEM_PROMPT = (
    "Eres un analista dermatologico independiente. Evalua una respuesta "
    "candidata producida por otro modelo usando solo la imagen, la consulta y, "
    "cuando existan, los casos recuperados. No cambies la respuesta candidata, "
    "no uses una respuesta de referencia y no describas una cadena de "
    "pensamiento. Entrega una justificacion clinica observable y prudente."
)


def load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero "
            "python3 -m src.build_contrastive_explanation_sample"
        )
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_contexts(record: dict[str, Any]) -> list[dict[str, Any]]:
    raw = record.get("rag_context_es", "")
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            value = json.loads(raw)
            return value if isinstance(value, list) else []
        except json.JSONDecodeError:
            return []
    return []


def build_explanation_messages(
    record: dict[str, Any],
    image_path: str,
    method: str,
    minimum_tokens: int,
    retry: bool = False,
    candidate_answer: str = "",
) -> list[dict[str, Any]]:
    use_rag = method in RAG_METHODS
    contexts = parse_contexts(record) if use_rag else []
    question = clean_text(record.get("question_es", ""))
    if use_rag:
        question = format_rag_prompt(question, contexts)

    retry_instruction = ""
    if retry:
        retry_instruction = (
            "\nTu salida anterior no cumplio el esquema o fue demasiado corta. "
            "Corrigela sin agregar informacion clinica no observable."
        )

    candidate_instruction = ""
    system_prompt = SYSTEM_PROMPT
    if candidate_answer:
        system_prompt = EXTERNAL_SYSTEM_PROMPT
        candidate_instruction = f"""

Respuesta candidata producida por el metodo evaluado:
{candidate_answer}

No la reescribas. Copiala exactamente en `answer_es` y evalua si esta
respaldada por la evidencia disponible.
""".rstrip()

    schema_instruction = f"""
{candidate_instruction}

Devuelve solamente un objeto JSON valido con este esquema:
{{
  "answer_es": "respuesta final al paciente",
  "explanation": "justificacion clinica observable de entre {minimum_tokens + 20} y {minimum_tokens + 80} tokens, nunca menos de {minimum_tokens}",
  "candidate_support": "supported, partially_supported, unsupported o uncertain",
  "visual_evidence_es": ["hallazgo visual 1", "hallazgo visual 2"],
  "question_evidence_es": ["dato explicito relevante de la consulta"],
  "uncertainty_es": "limitaciones o diagnosticos diferenciales prudentes",
  "rag_context_use_es": "como influyeron los casos recuperados, o no_aplica"
}}

La explicacion debe conectar evidencia y respuesta, no narrar pensamientos
internos. Si la imagen o la consulta no permiten sostener una conclusion,
declara esa limitacion. No recibes ni debes intentar reconstruir la respuesta
de referencia.{retry_instruction}
""".strip()

    content: list[dict[str, Any]] = [{"type": "image", "image": image_path}]
    content.append({"type": "text", "text": f"{question}\n\n{schema_instruction}"})
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content},
    ]


def extract_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("La respuesta no contiene un objeto JSON")
        parsed = json.loads(candidate[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("La respuesta JSON no es un objeto")
    return parsed


def normalize_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [clean_text(item) for item in value if clean_text(item)]
    text = clean_text(value)
    return [text] if text else []


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    answer = clean_text(payload.get("answer_es", ""))
    explanation = clean_text(
        payload.get("explanation", payload.get("explanation_es", ""))
    )
    if not answer:
        raise ValueError("Falta answer_es")
    if not explanation:
        raise ValueError("Falta explanation")
    return {
        "answer_es": answer,
        "explanation": explanation,
        "candidate_support": clean_text(payload.get("candidate_support", "")),
        "visual_evidence_es": normalize_string_list(
            payload.get("visual_evidence_es", [])
        ),
        "question_evidence_es": normalize_string_list(
            payload.get("question_evidence_es", [])
        ),
        "uncertainty_es": clean_text(payload.get("uncertainty_es", "")),
        "rag_context_use_es": clean_text(
            payload.get("rag_context_use_es", "no_aplica")
        ),
    }


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def token_f1(first: str, second: str) -> float:
    first_tokens = clean_text(first).lower().split()
    second_tokens = clean_text(second).lower().split()
    if not first_tokens or not second_tokens:
        return 0.0
    first_counts: dict[str, int] = {}
    second_counts: dict[str, int] = {}
    for token in first_tokens:
        first_counts[token] = first_counts.get(token, 0) + 1
    for token in second_tokens:
        second_counts[token] = second_counts.get(token, 0) + 1
    overlap = sum(
        min(count, second_counts.get(token, 0))
        for token, count in first_counts.items()
    )
    precision = overlap / len(first_tokens)
    recall = overlap / len(second_tokens)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def load_original_prediction_lookup(
    dataset: str, method: str
) -> dict[tuple[str, str], str]:
    path = ORIGINAL_PREDICTION_PATHS.get((dataset, method))
    if not path or not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            (
                clean_text(row.get("encounter_id", "")),
                clean_text(row.get("image_id", "")),
            ): clean_text(row.get("predicted_answer_es", ""))
            for row in csv.DictReader(handle)
        }


def output_path(output_dir: Path, method: str) -> Path:
    return output_dir / f"explanations_{method}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-infiere casos contrastivos con explicacion estructurada."
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_NAMES),
        default="enriched",
    )
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--cases", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--quantize", choices=["4bit", "8bit"], default="4bit")
    parser.add_argument("--min-explanation-tokens", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=640)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument(
        "--explanation-mode",
        choices=("self", "external"),
        default="self",
        help="self re-infiere; external explica la prediccion original.",
    )
    parser.add_argument(
        "--resume-failed",
        action="store_true",
        help="Reprocesa solo filas invalidas del CSV existente y las reemplaza.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.min_explanation_tokens < 1:
        raise ValueError("--min-explanation-tokens debe ser positivo")
    if (
        args.explanation_mode == "self"
        and args.method.startswith("lora")
        and not args.adapter
        and not args.dry_run
    ):
        raise ValueError(f"{args.method} requiere --adapter")

    dataset_name = DATASET_NAMES[args.dataset]
    cases_path = args.cases or (
        PROJECT_ROOT
        / "outputs"
        / "error_analysis"
        / dataset_name
        / "contrastive_cases_test.jsonl"
    )
    output_subdir = (
        "explanations"
        if args.explanation_mode == "self"
        else "external_rationales"
    )
    output_dir = args.output_dir or (
        PROJECT_ROOT / "outputs" / "error_analysis" / dataset_name / output_subdir
    )
    path = output_path(output_dir, args.method)
    cases = load_cases(cases_path)
    if args.limit:
        cases = cases[: args.limit]

    resolver = ImageResolver()
    resolved: list[tuple[dict[str, Any], str]] = []
    missing: list[str] = []
    for record in cases:
        image_path, missing_id = resolver.resolve(record)
        if image_path:
            resolved.append((record, image_path))
        else:
            missing.append(missing_id or clean_text(record.get("image_id", "")))

    existing_rows: list[dict[str, Any]] = []
    if args.resume_failed and path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            existing_rows = list(csv.DictReader(handle))
        failed_keys = {
            (
                clean_text(row.get("encounter_id", "")),
                clean_text(row.get("image_id", "")),
            )
            for row in existing_rows
            if row.get("parse_error", "")
            or str(row.get("meets_minimum_tokens", "")).lower()
            not in {"true", "1"}
        }
        resolved = [
            (record, image_path)
            for record, image_path in resolved
            if (
                clean_text(record.get("encounter_id", "")),
                clean_text(record.get("image_id", "")),
            )
            in failed_keys
        ]

    print(f"Dataset: {dataset_name}")
    print(f"Metodo: {args.method}")
    print(f"Modo: {args.explanation_mode}")
    print(f"Casos seleccionados: {len(cases)}")
    if args.resume_failed:
        print(f"Casos invalidos a reparar: {len(resolved)}")
    print(f"Imagenes resueltas: {len(resolved)} | faltantes: {len(missing)}")
    if missing:
        print("Primeras faltantes:", ", ".join(missing[:5]))
    if not resolved:
        if args.resume_failed:
            print("No quedan filas invalidas para reparar.")
            return 0
        raise FileNotFoundError(
            "No se resolvio ninguna imagen para los casos seleccionados."
        )

    if args.dry_run:
        if resolved:
            record = resolved[0][0]
            key = (
                clean_text(record.get("encounter_id", "")),
                clean_text(record.get("image_id", "")),
            )
            lookup = load_original_prediction_lookup(args.dataset, args.method)
            candidate = (
                lookup.get(key, "")
                if args.explanation_mode == "external"
                else ""
            )
            messages = build_explanation_messages(
                record,
                resolved[0][1],
                args.method,
                args.min_explanation_tokens,
                candidate_answer=candidate,
            )
            print(json.dumps(messages, ensure_ascii=False, indent=2)[:4000])
        return 0

    source_adapter = str(args.adapter) if args.adapter else ""
    adapter = source_adapter if args.explanation_mode == "self" else None
    model, processor = load_model_and_processor(
        args.model, args.quantize, adapter
    )
    rows: list[dict[str, Any]] = []
    original_column = f"{args.method}_predicted_answer_es"
    original_lookup = load_original_prediction_lookup(args.dataset, args.method)

    for index, (record, image_path) in enumerate(resolved, start=1):
        key = (
            clean_text(record.get("encounter_id", "")),
            clean_text(record.get("image_id", "")),
        )
        original_prediction = clean_text(
            record.get(original_column, "")
        ) or original_lookup.get(key, "")
        if args.explanation_mode == "external" and not original_prediction:
            raise ValueError(
                f"No hay prediccion original para {args.method} {key}"
            )

        payload: dict[str, Any] | None = None
        raw_output = ""
        parse_error = ""
        explanation_tokens = 0
        attempts = 0
        generation_time_s = 0.0
        for attempt in range(args.max_retries + 1):
            attempts = attempt + 1
            messages = build_explanation_messages(
                record,
                image_path,
                args.method,
                args.min_explanation_tokens,
                retry=attempt > 0,
                candidate_answer=(
                    original_prediction
                    if args.explanation_mode == "external"
                    else ""
                ),
            )
            generation_start = time.perf_counter()
            raw_output = generate_answer(
                model, processor, messages, args.max_new_tokens
            )
            generation_time_s += time.perf_counter() - generation_start
            try:
                parsed = extract_json_object(raw_output)
                if args.explanation_mode == "external":
                    parsed.setdefault("answer_es", original_prediction)
                payload = normalize_payload(parsed)
                explanation_tokens = token_count(
                    processor.tokenizer, payload["explanation"]
                )
                if explanation_tokens >= args.min_explanation_tokens:
                    parse_error = ""
                    break
                parse_error = (
                    f"explanation demasiado corta: {explanation_tokens} "
                    f"< {args.min_explanation_tokens} tokens"
                )
            except (ValueError, json.JSONDecodeError) as error:
                payload = None
                parse_error = str(error)

        payload = payload or {
            "answer_es": "",
            "explanation": "",
            "candidate_support": "",
            "visual_evidence_es": [],
            "question_evidence_es": [],
            "uncertainty_es": "",
            "rag_context_use_es": "",
        }
        if args.explanation_mode == "external":
            payload["answer_es"] = original_prediction
        rows.append(
            {
                "dataset_variant": dataset_name,
                "method": args.method,
                "explanation_mode": args.explanation_mode,
                "model_name": args.model,
                "adapter_path": source_adapter,
                "encounter_id": clean_text(record.get("encounter_id", "")),
                "image_id": clean_text(record.get("image_id", "")),
                "image_path": image_path,
                "selection_buckets": clean_text(
                    record.get("selection_buckets", "")
                ),
                "question_es": clean_text(record.get("question_es", "")),
                "reference_answer_es": clean_text(
                    record.get("reference_answer_es", "")
                ),
                "original_prediction_es": original_prediction,
                "reanalysis_answer_es": payload["answer_es"],
                "explanation": payload["explanation"],
                "candidate_support": payload["candidate_support"],
                "visual_evidence_es": json.dumps(
                    payload["visual_evidence_es"], ensure_ascii=False
                ),
                "question_evidence_es": json.dumps(
                    payload["question_evidence_es"], ensure_ascii=False
                ),
                "uncertainty_es": payload["uncertainty_es"],
                "rag_context_use_es": payload["rag_context_use_es"],
                "retrieved_encounter_ids": clean_text(
                    record.get("retrieved_encounter_ids", "")
                ),
                "retrieved_scores": clean_text(
                    record.get("retrieved_scores", "")
                ),
                "rag_context_es": json.dumps(
                    parse_contexts(record) if args.method in RAG_METHODS else [],
                    ensure_ascii=False,
                ),
                "explanation_token_count": explanation_tokens,
                "minimum_explanation_tokens": args.min_explanation_tokens,
                "max_new_tokens": args.max_new_tokens,
                "meets_minimum_tokens": (
                    explanation_tokens >= args.min_explanation_tokens
                ),
                "original_vs_reanalysis_token_f1": token_f1(
                    original_prediction, payload["answer_es"]
                ),
                "attempts": attempts,
                "generation_time_s": generation_time_s,
                "parse_error": parse_error,
                "raw_output": raw_output,
            }
        )
        if index % 5 == 0 or index == len(resolved):
            print(f"  {index}/{len(resolved)} explicaciones generadas")

    if existing_rows:
        replacement_lookup = {
            (row["encounter_id"], row["image_id"]): row for row in rows
        }
        rows = [
            replacement_lookup.get(
                (
                    clean_text(row.get("encounter_id", "")),
                    clean_text(row.get("image_id", "")),
                ),
                row,
            )
            for row in existing_rows
        ]

    output_dir.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    valid = sum(
        bool(row["meets_minimum_tokens"]) and not row["parse_error"]
        for row in rows
    )
    print(f"Salida: {path}")
    print(f"JSON valido y longitud cumplida: {valid}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
