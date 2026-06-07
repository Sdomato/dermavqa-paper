"""Build a DermaVQA dataset with one LLM-synthesized answer per case.

The synthesis prompt is strictly text-only: question + source answers. Image
paths are kept only in the final dataset rows for later multimodal fine-tuning.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is declared in requirements.
    load_dotenv = None

try:
    from openai import APIConnectionError, APIStatusError, AzureOpenAI, BadRequestError, OpenAI, RateLimitError
except ImportError:  # pragma: no cover - imported only when Azure calls run.
    APIConnectionError = Exception
    APIStatusError = Exception
    AzureOpenAI = None
    OpenAI = None
    BadRequestError = Exception
    RateLimitError = Exception

try:
    from pydantic import BaseModel, Field, ValidationError
except ImportError:  # pragma: no cover - dry-run still works with manual checks.
    BaseModel = None
    Field = None
    ValidationError = ValueError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_VERSION = "llm_synthesis_es_v5"
DEFAULT_API_VERSION = "2024-10-21"
DEFAULT_DEPLOYMENT = "gpt-oss-120b"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 900
MIN_SYNTHESIS_WORDS = 5
SPLIT_FILES = {
    "train": "train.json",
    "valid": "valid_ht.json",
    "test": "test_ht_spanishtestsetcorrected.json",
}


if BaseModel is not None:

    class SynthesisResult(BaseModel):
        synthesized_answer_es: str = Field(min_length=1)
        has_conflict: bool
        conflict_note: str
        source_support_level: str


else:
    SynthesisResult = None


SYNTHESIS_RESPONSE_SCHEMA: dict[str, Any] = {
    "name": "dermavqa_synthesis",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "synthesized_answer_es": {
                "type": "string",
                "description": "Respuesta clinica concisa en espanol sustentada solo en la informacion de entrada.",
            },
            "has_conflict": {
                "type": "boolean",
                "description": "True si las respuestas originales se contradicen entre si.",
            },
            "conflict_note": {
                "type": "string",
                "description": "Nota breve sobre contradicciones; cadena vacia si no aplica.",
            },
            "source_support_level": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "Nivel de soporte de la sintesis segun cantidad y consistencia de respuestas originales.",
            },
        },
        "required": [
            "synthesized_answer_es",
            "has_conflict",
            "conflict_note",
            "source_support_level",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a text-only LLM-synthesized DermaVQA dataset."
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(SPLIT_FILES),
        default=["train", "valid", "test"],
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit cases, not image rows.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write outputs.")
    parser.add_argument(
        "--call-azure",
        action="store_true",
        help="In dry-run mode, call Azure but still do not write outputs or cache.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--response-format",
        choices=["auto", "json_schema", "json_object"],
        default="auto",
        help="Use json_schema when supported; auto retries with json_object on unsupported deployments.",
    )
    parser.add_argument("--output-prefix", default="dermavqa_iiyi_llm_synthesized_by_image")
    parser.add_argument(
        "--deployment",
        default=os.environ.get("AZURE_OPENAI_DEPLOYMENT", DEFAULT_DEPLOYMENT),
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "azure_openai", "foundry_v1"],
        default=os.environ.get("LLM_PROVIDER", "auto"),
        help="Use foundry_v1 for Azure AI Foundry models such as gpt-oss-120b.",
    )
    parser.add_argument(
        "--api-version",
        default=os.environ.get("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION),
    )
    return parser.parse_args()


def load_config() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        return {"data_dir": "data/iiyi", "output_dir": "outputs"}
    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    return loaded


def resolve_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_answer(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def dedupe_answers(answers: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for answer in answers:
        clean = str(answer).strip()
        if not clean:
            continue
        normalized = normalize_answer(clean)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(clean)
    return deduped


def build_question(case: dict[str, Any]) -> str:
    parts = [
        str(case.get("query_title_es") or "").strip(),
        str(case.get("query_content_es") or "").strip(),
    ]
    return "\n".join(part for part in parts if part)


def get_source_answers(case: dict[str, Any]) -> list[str]:
    responses = case.get("responses") or []
    answers: list[str] = []
    for response in responses:
        if isinstance(response, dict):
            answer = response.get("content_es")
            if answer:
                answers.append(str(answer).strip())
    return answers


def source_hash(question_es: str, source_answers_es: list[str]) -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "question_es": question_es,
        "source_answers_es": source_answers_es,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_cases(data_dir: Path, splits: list[str]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for split in splits:
        path = data_dir / SPLIT_FILES[split]
        with path.open("r", encoding="utf-8") as file:
            split_cases = json.load(file)
        for case in split_cases:
            case["split"] = split
            cases.append(case)
    return cases


def build_image_index(images_dir: Path) -> dict[str, Path]:
    if not images_dir.exists():
        return {}
    return {path.name: path for path in images_dir.rglob("*") if path.is_file()}


def build_messages(question_es: str, source_answers_es: list[str]) -> list[dict[str, str]]:
    numbered_answers = "\n".join(
        f"{index}. {answer}" for index, answer in enumerate(source_answers_es, start=1)
    )
    system_prompt = (
        "Sos un redactor clinico para un dataset de VQA dermatologico en espanol. "
        "Tu tarea es unificar respuestas medicas existentes en una sola respuesta "
        "clara, prudente y completa. No ves imagenes y no debes inferir nada fuera "
        "de la pregunta y las respuestas originales. Devolve exclusivamente un objeto JSON valido."
    )
    user_prompt = f"""Pregunta del paciente:
{question_es}

Respuestas medicas originales deduplicadas:
{numbered_answers}

Instrucciones:
- Escribi una unica respuesta en espanol.
- Estilo: clinica concisa.
- Longitud objetivo: 60 a 100 palabras si la informacion disponible lo permite.
- Usa solo informacion presente en las respuestas originales.
- No agregues diagnosticos, tratamientos, advertencias ni recomendaciones nuevas.
- No agregues morfologia, signos negativos, causas, relaciones clinicas ni hallazgos no mencionados explicitamente.
- Nunca devuelvas una respuesta de una sola palabra o solo una etiqueta.
- Si las respuestas originales solo dan una etiqueta diagnostica, conviertela en una frase natural de asistente medico, por ejemplo: "El cuadro es compatible con [diagnostico]."
- No uses certeza absoluta como "claramente" salvo que esa certeza aparezca explicitamente en las respuestas originales.
- No expandas siglas o abreviaturas medicas si la fuente no las desarrolla.
- Si la informacion disponible es muy breve, responde brevemente sin inflarla.
- Si hay contradicciones, integra la incertidumbre de forma breve.
- Marca has_conflict como true solo si hay contradiccion directa sobre el mismo diagnostico o hallazgo.
- No marques conflicto si una respuesta agrega contexto adicional sin contradecir el diagnostico principal.
- Redacta como si fuera tu propia evaluacion clinica, no como un resumen de opiniones ajenas.
- Cuando haya varias posibilidades, usa frases clinicas directas como "El cuadro es compatible con..." o "Entre los diagnosticos diferenciales se consideran...".
- No uses frases meta como "las respuestas", "las fuentes", "los autores", "diagnosticos propuestos", "se sugiere en las respuestas", "discrepancias entre..." ni similares.
- No menciones que estas unificando respuestas ni que sos un modelo.
"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def validate_synthesis(payload: dict[str, Any]) -> dict[str, Any]:
    for wrapper_key in ("final", "output", "result"):
        wrapped = payload.get(wrapper_key)
        if isinstance(wrapped, dict):
            outer_payload = {key: value for key, value in payload.items() if key != wrapper_key}
            payload = {**outer_payload, **wrapped}
            break
        if isinstance(wrapped, str):
            try:
                nested = json.loads(wrapped)
            except json.JSONDecodeError:
                continue
            if isinstance(nested, dict):
                outer_payload = {key: value for key, value in payload.items() if key != wrapper_key}
                payload = {**outer_payload, **nested}
                break

    if "source_support_level" in payload and payload["source_support_level"] not in {
        "low",
        "medium",
        "high",
    }:
        payload["source_support_level"] = "medium"
    if payload.get("conflict_note") is None:
        payload["conflict_note"] = ""

    if SynthesisResult is not None:
        try:
            result = SynthesisResult.model_validate(payload)
        except ValidationError as error:
            raise ValueError(f"Invalid synthesis payload: {error}") from error
        payload = result.model_dump()
    required = {
        "synthesized_answer_es": str,
        "has_conflict": bool,
        "conflict_note": str,
        "source_support_level": str,
    }
    for key, expected_type in required.items():
        if key not in payload:
            raise ValueError(f"Missing synthesis field: {key}")
        if not isinstance(payload[key], expected_type):
            raise ValueError(f"Invalid type for {key}: expected {expected_type.__name__}")
    payload["synthesized_answer_es"] = payload["synthesized_answer_es"].strip()
    payload["conflict_note"] = payload["conflict_note"].strip()
    if not payload["synthesized_answer_es"]:
        raise ValueError("synthesized_answer_es cannot be empty")
    word_count = len(re.findall(r"\b\w+\b", payload["synthesized_answer_es"]))
    if word_count < MIN_SYNTHESIS_WORDS:
        raise ValueError(
            f"synthesized_answer_es is too short: {word_count} words, "
            f"minimum is {MIN_SYNTHESIS_WORDS}"
        )
    if payload["source_support_level"] not in {"low", "medium", "high"}:
        raise ValueError("source_support_level must be low, medium, or high")
    return payload


def extract_message_content(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif isinstance(part.get("content"), str):
                    parts.append(part["content"])
            else:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return ""


def parse_json_payload(content: str) -> dict[str, Any]:
    stripped = content.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        fenced = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        fenced = re.sub(r"\s*```$", "", fenced)
        try:
            payload = json.loads(fenced)
        except json.JSONDecodeError:
            start = fenced.find("{")
            end = fenced.rfind("}")
            if start < 0 or end <= start:
                raise
            payload = json.loads(fenced[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Synthesis payload must be a JSON object")
    return payload


def mock_synthesis(source_answers_es: list[str]) -> dict[str, Any]:
    if not source_answers_es:
        answer = "No hay respuestas medicas disponibles para sintetizar."
        support = "low"
    elif len(source_answers_es) == 1:
        answer = f"El cuadro es compatible con {source_answers_es[0].rstrip('.')}."
        support = "low"
    else:
        answer = " ".join(source_answers_es[:4])
        support = "medium" if len(source_answers_es) < 5 else "high"
    return validate_synthesis(
        {
            "synthesized_answer_es": answer,
            "has_conflict": False,
            "conflict_note": "",
            "source_support_level": support,
        }
    )


def normalize_foundry_base_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/openai/v1"):
        return endpoint + "/"
    return endpoint + "/openai/v1/"


def should_use_foundry_v1(provider: str, endpoint: str) -> bool:
    if provider == "foundry_v1":
        return True
    if provider == "azure_openai":
        return False
    normalized = endpoint.lower().rstrip("/")
    return "services.ai.azure.com" in normalized or normalized.endswith("/openai/v1")


def make_llm_client(api_version: str, provider: str) -> tuple[Any, str]:
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")
    endpoint = (
        os.environ.get("AZURE_OPENAI_ENDPOINT")
        or os.environ.get("AZURE_AI_FOUNDRY_ENDPOINT")
        or os.environ.get("AZURE_INFERENCE_ENDPOINT")
    )
    api_key = (
        os.environ.get("AZURE_OPENAI_API_KEY")
        or os.environ.get("AZURE_AI_FOUNDRY_KEY")
        or os.environ.get("AZURE_INFERENCE_CREDENTIAL")
    )
    if not endpoint or not api_key:
        raise RuntimeError(
            "Missing Azure credentials. Set AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_API_KEY "
            "or AZURE_AI_FOUNDRY_ENDPOINT/AZURE_AI_FOUNDRY_KEY."
        )
    if should_use_foundry_v1(provider, endpoint):
        if OpenAI is None:
            raise RuntimeError("Missing dependency: install openai>=1.42.0")
        return OpenAI(api_key=api_key, base_url=normalize_foundry_base_url(endpoint)), "foundry_v1"
    if AzureOpenAI is None:
        raise RuntimeError("Missing dependency: install openai>=1.42.0")
    return (
        AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version),
        "azure_openai",
    )


def call_azure(
    client: Any,
    deployment: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    response_format: str,
) -> tuple[dict[str, Any], str | None]:
    json_retry_instruction = (
        "La respuesta anterior no fue JSON valido. Reintenta ahora "
        "devolviendo exclusivamente un objeto JSON valido con estas "
        "claves: synthesized_answer_es, has_conflict, conflict_note, "
        "source_support_level. No agregues texto fuera del JSON."
    )
    plain_retry_instruction = (
        "Reintenta sin usar ningun formato especial de API. Devolve solo un "
        "objeto JSON, sin markdown ni texto adicional. El objeto debe incluir "
        "exactamente estas claves: synthesized_answer_es, has_conflict, "
        "conflict_note, source_support_level."
    )
    compact_retry_instruction = (
        "Si el caso anterior te resulta ambiguo, redacta una respuesta clinica "
        "breve y prudente usando solo las respuestas originales. Devolve solo "
        "JSON valido con synthesized_answer_es, has_conflict, conflict_note y "
        "source_support_level."
    )
    first_format = "json_schema" if response_format in {"auto", "json_schema"} else "json_object"
    attempts: list[tuple[str, str | None, str | None, int]] = [
        ("structured", first_format, None, max_tokens),
        ("json_retry", "json_object", json_retry_instruction, max(max_tokens, 1600)),
        ("plain_retry", None, plain_retry_instruction, max(max_tokens, 1600)),
        ("compact_plain_retry", None, compact_retry_instruction, max(max_tokens, 1600)),
    ]
    last_error: Exception | None = None

    for attempt_name, format_kind, retry_instruction, token_limit in attempts:
        request_messages = messages
        if retry_instruction is not None:
            request_messages = messages + [{"role": "user", "content": retry_instruction}]
        request: dict[str, Any] = {
            "model": deployment,
            "messages": request_messages,
            "temperature": temperature,
            "max_tokens": token_limit,
        }
        if format_kind == "json_schema":
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": SYNTHESIS_RESPONSE_SCHEMA,
            }
        elif format_kind == "json_object":
            request["response_format"] = {"type": "json_object"}

        for transient_attempt in range(5):
            try:
                response = client.chat.completions.create(**request)
                break
            except BadRequestError as error:
                last_error = error
                if response_format != "auto" or format_kind != "json_schema":
                    raise
                request["response_format"] = {"type": "json_object"}
                response = client.chat.completions.create(**request)
                break
            except (APIConnectionError, APIStatusError, RateLimitError) as error:
                last_error = error
                if transient_attempt == 4:
                    raise
                retry_after = getattr(error, "response", None)
                retry_after_header = None
                if retry_after is not None:
                    retry_after_header = retry_after.headers.get("retry-after")
                wait_seconds = int(retry_after_header) if retry_after_header else min(
                    90, 10 * (transient_attempt + 1)
                )
                time.sleep(wait_seconds)

        message = response.choices[0].message
        content = extract_message_content(message)
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        if not content:
            refusal = getattr(message, "refusal", None)
            last_error = ValueError(
                "Azure returned an empty synthesis "
                f"on {attempt_name}. Finish reason: {finish_reason}. Refusal: {refusal}"
            )
            continue
        try:
            payload = parse_json_payload(content)
            return validate_synthesis(payload), finish_reason
        except (json.JSONDecodeError, ValueError) as error:
            last_error = error

    raise ValueError(f"Azure returned invalid JSON after retry: {last_error}")


def cache_path(cache_dir: Path, split: str, encounter_id: str) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", encounter_id)
    return cache_dir / f"{split}_{safe_id}.json"


def load_cached_result(
    path: Path,
    expected_hash: str,
    deployment: str,
    api_version: str,
    temperature: float,
    provider: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        cached = json.load(file)
    cache_matches = (
        cached.get("source_hash") == expected_hash
        and cached.get("synthesis_model") == deployment
        and cached.get("api_version") == api_version
        and cached.get("synthesis_provider") == provider
        and cached.get("prompt_version") == PROMPT_VERSION
        and float(cached.get("temperature")) == float(temperature)
    )
    if not cache_matches:
        return None
    return validate_synthesis(cached["result"])


def write_cache(
    path: Path,
    case: dict[str, Any],
    source_answers_es: list[str],
    distinct_source_answers_es: list[str],
    source_hash_value: str,
    result: dict[str, Any],
    deployment: str,
    api_version: str,
    temperature: float,
    provider: str,
    finish_reason: str | None,
) -> None:
    payload = {
        "encounter_id": case["encounter_id"],
        "split": case["split"],
        "synthesis_provider": provider,
        "synthesis_model": deployment,
        "api_version": api_version,
        "prompt_version": PROMPT_VERSION,
        "temperature": temperature,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_hash": source_hash_value,
        "question_es": build_question(case),
        "source_answers_es": source_answers_es,
        "distinct_source_answers_es": distinct_source_answers_es,
        "finish_reason": finish_reason,
        "result": result,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def build_rows_for_case(
    case: dict[str, Any],
    image_index: dict[str, Path],
    source_answers_es: list[str],
    distinct_source_answers_es: list[str],
    result: dict[str, Any],
    deployment: str,
    temperature: float,
    provider: str,
    created_at: str,
) -> list[dict[str, Any]]:
    image_ids = case.get("image_ids") or []
    question_es = build_question(case)
    rows: list[dict[str, Any]] = []
    for image_id in image_ids:
        image_path = image_index.get(image_id)
        rows.append(
            {
                "encounter_id": case["encounter_id"],
                "split": case["split"],
                "image_id": image_id,
                "image_path": str(image_path) if image_path else "",
                "question_es": question_es,
                "source_answers_es": source_answers_es,
                "distinct_source_answers_es": distinct_source_answers_es,
                "source_answer_count": len(source_answers_es),
                "distinct_source_answer_count": len(distinct_source_answers_es),
                "synthesized_answer_es": result["synthesized_answer_es"],
                "synthesis_model": deployment,
                "synthesis_provider": provider,
                "prompt_version": PROMPT_VERSION,
                "temperature": temperature,
                "created_at": created_at,
                "has_conflict": result["has_conflict"],
                "conflict_note": result["conflict_note"],
                "source_support_level": result["source_support_level"],
            }
        )
    return rows


def print_dry_run(case: dict[str, Any], messages: list[dict[str, str]], result: dict[str, Any]) -> None:
    print("=" * 88)
    print(f"{case['split']} / {case['encounter_id']}")
    print("-" * 88)
    for message in messages:
        print(f"[{message['role']}]\n{message['content']}\n")
    print("[parsed_output]")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def write_outputs(rows: list[dict[str, Any]], output_dir: Path, output_prefix: str, overwrite: bool) -> None:
    dataset_dir = output_dir / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = dataset_dir / f"{output_prefix}.jsonl"
    csv_path = dataset_dir / f"{output_prefix}.csv"
    existing_outputs = [path for path in (jsonl_path, csv_path) if path.exists()]
    if existing_outputs and not overwrite:
        existing = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(f"Output exists. Pass --overwrite to replace: {existing}")

    with jsonl_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_rows = []
    for row in rows:
        csv_row = row.copy()
        csv_row["source_answers_es"] = json.dumps(
            csv_row["source_answers_es"], ensure_ascii=False
        )
        csv_row["distinct_source_answers_es"] = json.dumps(
            csv_row["distinct_source_answers_es"], ensure_ascii=False
        )
        csv_rows.append(csv_row)
    fieldnames = list(csv_rows[0].keys()) if csv_rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Wrote {len(rows)} rows")
    print(f"JSONL: {jsonl_path}")
    print(f"CSV  : {csv_path}")


def ensure_outputs_can_be_written(output_dir: Path, output_prefix: str, overwrite: bool) -> None:
    dataset_dir = output_dir / "datasets"
    jsonl_path = dataset_dir / f"{output_prefix}.jsonl"
    csv_path = dataset_dir / f"{output_prefix}.csv"
    existing_outputs = [path for path in (jsonl_path, csv_path) if path.exists()]
    if existing_outputs and not overwrite:
        existing = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(f"Output exists. Pass --overwrite to replace: {existing}")


def main() -> int:
    args = parse_args()
    config = load_config()
    data_dir = resolve_path(args.data_dir or config.get("data_dir", "data/iiyi"))
    output_dir = resolve_path(args.output_dir or config.get("output_dir", "outputs"))

    if args.dry_run and args.limit is None:
        args.limit = 10
        print("Dry-run without --limit defaults to 10 cases.", file=sys.stderr)

    cases = load_cases(data_dir, args.splits)
    if args.limit is not None:
        cases = cases[: args.limit]

    if not args.dry_run:
        ensure_outputs_can_be_written(output_dir, args.output_prefix, args.overwrite)

    image_index = build_image_index(data_dir / "images_final")
    cache_dir = output_dir / "llm_synthesis" / "cache"
    client = None
    resolved_provider = "dry_run"
    should_call_azure = not args.dry_run or args.call_azure
    if should_call_azure:
        client, resolved_provider = make_llm_client(args.api_version, args.provider)

    all_rows: list[dict[str, Any]] = []
    created_at = datetime.now(timezone.utc).isoformat()
    for index, case in enumerate(cases, start=1):
        question_es = build_question(case)
        source_answers_es = get_source_answers(case)
        distinct_source_answers_es = dedupe_answers(source_answers_es)
        hash_value = source_hash(question_es, source_answers_es)
        messages = build_messages(question_es, distinct_source_answers_es)
        case_cache_path = cache_path(cache_dir, case["split"], case["encounter_id"])

        result = None
        finish_reason = None
        used_cache = False
        if not args.dry_run:
            result = load_cached_result(
                case_cache_path,
                hash_value,
                args.deployment,
                args.api_version,
                args.temperature,
                resolved_provider,
            )
            used_cache = result is not None
        if result is None:
            if should_call_azure:
                result, finish_reason = call_azure(
                    client=client,
                    deployment=args.deployment,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    response_format=args.response_format,
                )
            else:
                result = mock_synthesis(distinct_source_answers_es)

        if args.dry_run:
            print_dry_run(case, messages, result)
            continue

        if not used_cache:
            write_cache(
                case_cache_path,
                case,
                source_answers_es,
                distinct_source_answers_es,
                hash_value,
                result,
                args.deployment,
                args.api_version,
                args.temperature,
                resolved_provider,
                finish_reason,
            )
        all_rows.extend(
            build_rows_for_case(
                case=case,
                image_index=image_index,
                source_answers_es=source_answers_es,
                distinct_source_answers_es=distinct_source_answers_es,
                result=result,
                deployment=args.deployment,
                temperature=args.temperature,
                provider=resolved_provider,
                created_at=created_at,
            )
        )
        if index % 25 == 0 or index == len(cases):
            print(f"Processed {index}/{len(cases)} cases")

    if not args.dry_run:
        write_outputs(all_rows, output_dir, args.output_prefix, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
