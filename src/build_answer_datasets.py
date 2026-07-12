"""
Build dataset_longest_answer from the raw IIYI JSON splits.

One record per encounter. The "answer" field contains the Spanish response
whose text is the longest (by character count) among all responses for that
encounter. This is an intermediate artifact: build_longest_by_image_dataset.py
expands it to dataset_longest_answer_by_image, the canonical target used by
retrieval, zero-shot and LoRA.

Every encounter with at least one response is included. Ties are broken by
selecting the first qualifying response in list order.

Output
------
outputs/datasets/dataset_longest_answer.json
outputs/datasets/dataset_longest_answer.csv
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "iiyi"
OUT_DIR = ROOT / "outputs" / "datasets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS: list[str] = ["train.json", "valid_ht.json", "test_ht_spanishtestsetcorrected.json"]

# Fields to carry forward into the output records
KEEP_FIELDS: list[str] = [
    "encounter_id",
    "author_id",
    "image_ids",
    "query_title_es",
    "query_content_es",
    "query_title_en",
    "query_content_en",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_splits(data_dir: Path, split_names: list[str]) -> list[dict[str, Any]]:
    """Load and concatenate JSON split files into a single list of records."""
    records: list[dict[str, Any]] = []
    for name in split_names:
        path = data_dir / name
        if not path.exists():
            print(f"  [WARN] Split not found, skipping: {path}", file=sys.stderr)
            continue
        with path.open(encoding="utf-8") as fh:
            split_data = json.load(fh)
        split_tag = name.replace(".json", "")
        for rec in split_data:
            rec["_split"] = split_tag
        records.extend(split_data)
        print(f"  Loaded {len(split_data):>4} records from {name}")
    return records


def select_answer(responses: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the response with the longest content_es text, or None if empty."""
    valid = [r for r in responses if r.get("content_es", "").strip()]
    if not valid:
        return None
    return max(valid, key=lambda r: len(r["content_es"]))


def build_record(
    raw: dict[str, Any],
    chosen: dict[str, Any],
) -> dict[str, Any]:
    """Compose a flat output record from the raw encounter + chosen response."""
    record: dict[str, Any] = {field: raw.get(field) for field in KEEP_FIELDS}
    record["_split"] = raw.get("_split")
    record["answer_es"] = chosen["content_es"]
    record["answer_en"] = chosen.get("content_en", "")
    record["answer_author_id"] = chosen.get("author_id", "")
    record["n_responses"] = len(raw.get("responses", []))
    return record


def build_dataset(raw_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Iterate over raw encounters and build the longest-answer dataset."""
    dataset: list[dict[str, Any]] = []
    skipped = 0
    for raw in raw_records:
        responses = raw.get("responses", [])
        chosen = select_answer(responses)
        if chosen is None:
            skipped += 1
            continue
        dataset.append(build_record(raw, chosen))
    if skipped:
        print(f"  [INFO] Skipped {skipped} encounters with no valid Spanish response.")
    return dataset


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_json(data: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    print(f"  Saved JSON  -> {path.relative_to(ROOT)}")


def save_csv(data: list[dict[str, Any]], path: Path) -> None:
    if not data:
        return
    fieldnames = list(data[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"  Saved CSV   -> {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n=== Loading raw IIYI splits ===")
    raw_records = load_splits(DATA_DIR, SPLITS)
    print(f"  Total encounters loaded: {len(raw_records)}")

    print("\n=== Building dataset_longest_answer ===")
    longest_ds = build_dataset(raw_records)
    print(f"  Records: {len(longest_ds)}")
    save_json(longest_ds, OUT_DIR / "dataset_longest_answer.json")
    save_csv(longest_ds, OUT_DIR / "dataset_longest_answer.csv")

    # Quick sanity check: show length distribution of chosen answers
    lengths = [len(r["answer_es"]) for r in longest_ds]
    print(f"  answer_es length — min:{min(lengths)}  mean:{sum(lengths)//len(lengths)}  max:{max(lengths)}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
