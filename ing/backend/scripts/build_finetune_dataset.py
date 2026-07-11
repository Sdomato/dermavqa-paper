"""
Fase 4 — Exportar los casos aprobados como dataset para el reentrenamiento.

Toma el JSONL de casos aprobados (lo que los médicos validaron en producción) y
lo escribe en el **mismo formato de registro** que el dataset del paper, listo
para mergear al split de entrenamiento de `src/train_longest.py`.

Dos destinos de este dataset:
  • Retrieval  — los casos aprobados ya retroalimentan la base en caliente (el
                 servicio los reindexa al aprobarlos); este export no hace falta
                 para eso.
  • LoRA       — `src/train_longest.py` descarta los items SIN imágenes, así que
                 del retrain participan solo los casos aprobados con foto. Los de
                 solo texto quedan igual en el dataset humano-validado (valiosos
                 para retrieval y evaluación), pero no entran al fine-tuning del VLM.

Es un paso OFFLINE y sin GPU: solo transforma JSON. El entrenamiento en sí corre
aparte, en una VM con GPU (ver ing/docs/fase4-reentrenamiento.md).

Uso:
    python -m scripts.build_finetune_dataset \
        --aprobados ing/backend/.data/casos_aprobados.jsonl \
        --salida outputs/datasets/aprobados_train.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Any

# Split con el que se marcan los casos que vienen del loop de mejora, para
# poder distinguirlos del dataset original al auditar el dataset combinado.
SPLIT_RETRAIN = "train_aprobado"


def cargar_aprobados(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    casos: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            casos.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return casos


def a_registro_dataset(caso: dict[str, Any]) -> dict[str, Any]:
    """Caso aprobado → registro con las mismas claves que usa load_dataset()."""
    return {
        "encounter_id": caso["encounter_id"],
        "query_title_es": caso.get("consulta", ""),
        "query_content_es": "",
        "answer_es": caso.get("respuesta", ""),
        "image_ids": caso.get("image_ids", []) or [],
        "_split": SPLIT_RETRAIN,
        "_origen": "loop_mejora",
        "_revisor": caso.get("revisor"),
    }


def construir(aprobados: Path, salida: Path) -> dict[str, int]:
    casos = cargar_aprobados(aprobados)
    registros = [a_registro_dataset(c) for c in casos]

    salida.parent.mkdir(parents=True, exist_ok=True)
    with salida.open("w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    con_imagen = sum(1 for r in registros if r["image_ids"])
    return {
        "total": len(registros),
        "con_imagen_para_lora": con_imagen,
        "solo_texto": len(registros) - con_imagen,
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    p = argparse.ArgumentParser(description="Exportar casos aprobados como dataset de retrain (Fase 4)")
    p.add_argument(
        "--aprobados", type=Path,
        default=repo_root / "ing" / "backend" / ".data" / "casos_aprobados.jsonl",
        help="JSONL de casos aprobados (default: el del servicio)",
    )
    p.add_argument(
        "--salida", type=Path,
        default=repo_root / "outputs" / "datasets" / "aprobados_train.jsonl",
        help="JSONL de salida en formato de registro del dataset",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stats = construir(args.aprobados, args.salida)
    print(f"Casos aprobados exportados: {stats['total']}")
    print(f"  · con imagen (entran al retrain LoRA): {stats['con_imagen_para_lora']}")
    print(f"  · solo texto (dataset humano-validado): {stats['solo_texto']}")
    print(f"Salida: {args.salida}")


if __name__ == "__main__":
    main()
