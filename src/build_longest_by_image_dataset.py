"""
Build a by-image version of dataset_longest_answer.

The original `dataset_longest_answer` has one row per encounter and may contain
multiple images in `image_ids`. This script expands each encounter into one row
per image, matching the interface used by the enriched dataset:

  split, encounter_id, image_id, image_path, question_es, answer_es

Outputs:
  outputs/datasets/dataset_longest_answer_by_image.json
  outputs/datasets/dataset_longest_answer_by_image.jsonl
  outputs/datasets/dataset_longest_answer_by_image.csv
  outputs/datasets/dataset_longest_answer_by_image.zip
"""

from __future__ import annotations

import csv
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from src.retrieval_utils import PROJECT_ROOT, build_query_text, clean_text, find_image, load_dataset

DATASETS_DIR = PROJECT_ROOT / "outputs" / "datasets"
SOURCE_PATH = DATASETS_DIR / "dataset_longest_answer.json"
OUT_PREFIX = DATASETS_DIR / "dataset_longest_answer_by_image"

SPLIT_MAP = {
    "train": "train",
    "valid_ht": "valid",
    "test_ht_spanishtestsetcorrected": "test",
}

FIELDNAMES = [
    "split",
    "encounter_id",
    "image_id",
    "image_path",
    "question_es",
    "answer_es",
]


def project_relative(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def expand_record(record: dict[str, Any]) -> list[dict[str, str]]:
    split = SPLIT_MAP.get(str(record.get("_split", "")), str(record.get("_split", "")))
    question_es = build_query_text(record)
    answer_es = clean_text(record.get("answer_es", ""))
    image_ids = [clean_text(image_id) for image_id in (record.get("image_ids") or [])]
    image_ids = [image_id for image_id in image_ids if image_id]

    rows: list[dict[str, str]] = []
    for image_id in image_ids:
        image_path = find_image(image_id)
        rows.append(
            {
                "split": split,
                "encounter_id": clean_text(record.get("encounter_id", "")),
                "image_id": image_id,
                "image_path": project_relative(image_path),
                "question_es": question_es,
                "answer_es": answer_es,
            }
        )
    return rows


def build_dataset(source_path: Path = SOURCE_PATH) -> list[dict[str, str]]:
    records = load_dataset(source_path)
    rows: list[dict[str, str]] = []
    skipped_no_images = 0
    skipped_no_answer = 0

    for record in records:
        if not clean_text(record.get("answer_es", "")):
            skipped_no_answer += 1
            continue
        expanded = expand_record(record)
        if not expanded:
            skipped_no_images += 1
            continue
        rows.extend(expanded)

    if skipped_no_images or skipped_no_answer:
        print(
            "Skipped encounters -> "
            f"no_images={skipped_no_images}, no_answer={skipped_no_answer}"
        )
    return rows


def write_json(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)


def write_jsonl(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(rows: list[dict[str, str]], path: Path) -> None:
    split_counts = Counter(row["split"] for row in rows)
    payload = {
        "dataset": "dataset_longest_answer_by_image",
        "source": str(SOURCE_PATH.relative_to(PROJECT_ROOT)),
        "description": (
            "Expanded by-image version of dataset_longest_answer. "
            "Each image from an encounter becomes one training row with the same "
            "Spanish question and longest-answer target."
        ),
        "row_count": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "columns": FIELDNAMES,
        "training_input": "image + question_es",
        "training_target": "answer_es",
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_readme(path: Path) -> None:
    path.write_text(
        "# dataset_longest_answer_by_image\n\n"
        "Version by-image del dataset `dataset_longest_answer`.\n\n"
        "- Input de entrenamiento: imagen + `question_es`.\n"
        "- Target: `answer_es`, elegido como la respuesta original mas larga del caso.\n"
        "- Si un caso tiene varias imagenes, se genera una fila por imagen.\n"
        "- Las imagenes no se incluyen en el zip; se resuelven por `image_id` desde "
        "`data/iiyi/images_final/`.\n",
        encoding="utf-8",
    )


def write_zip(files: list[Path], zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in files:
            archive.write(file_path, arcname=file_path.name)


def main() -> int:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_dataset()
    rows.sort(key=lambda row: (row["split"], row["encounter_id"], row["image_id"]))

    json_path = OUT_PREFIX.with_suffix(".json")
    jsonl_path = OUT_PREFIX.with_suffix(".jsonl")
    csv_path = OUT_PREFIX.with_suffix(".csv")
    manifest_path = DATASETS_DIR / "dataset_longest_answer_by_image_manifest.json"
    readme_path = DATASETS_DIR / "dataset_longest_answer_by_image_README.md"
    zip_path = OUT_PREFIX.with_suffix(".zip")

    write_json(rows, json_path)
    write_jsonl(rows, jsonl_path)
    write_csv(rows, csv_path)
    write_manifest(rows, manifest_path)
    write_readme(readme_path)
    write_zip([json_path, jsonl_path, csv_path, manifest_path, readme_path], zip_path)

    split_counts = Counter(row["split"] for row in rows)
    missing_paths = sum(1 for row in rows if not row["image_path"])
    print(f"Rows: {len(rows)}")
    print(f"Split counts: {dict(sorted(split_counts.items()))}")
    print(f"Rows with unresolved image_path: {missing_paths}")
    print(f"Wrote: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote: {zip_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
