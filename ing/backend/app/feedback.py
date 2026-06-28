"""
Loop de mejora (Fase 4): casos aprobados que retroalimentan la base.

Cuando un médico aprueba (o edita y aprueba) un borrador, esa consulta + la
respuesta validada se guardan como un **caso nuevo**. Esos casos se suman a la
base buscable, así que una consulta parecida en el futuro los recupera como
evidencia. Es el mecanismo por el que *cada aprobación mejora el sistema*.

Dos roles, sobre el mismo archivo JSONL append-only:
  1. Retroalimentar el retrieval (`to_cases()` → se reindexan junto al corpus).
  2. Acumular el **dataset de validación clínica humana real** (lo que al paper
     le falta): cada línea es un par consulta→respuesta aprobado por un médico.

En producción esto iría a una base con trazabilidad; un JSONL alcanza para el
MVP y es inspeccionable a mano (igual que el audit log de la Fase 3).
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .retrieval.corpus import Case

# Split sintético con el que se marca un caso que vino del loop de mejora
# (no del dataset original). Permite distinguirlos en los resultados.
SPLIT_APROBADO = "aprobado"


class CasosAprobadosStore:
    """Almacén append-only de casos aprobados por un médico."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._entries: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    self._entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    def agregar(
        self,
        consulta: str,
        respuesta: str,
        revisor: str | None = None,
        job_id: str | None = None,
        image_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Persistir un caso aprobado y devolver su registro (con id y timestamp)."""
        entry = {
            "encounter_id": "APR-" + uuid.uuid4().hex[:10],
            "timestamp": datetime.now(UTC).isoformat(),
            "consulta": consulta,
            "respuesta": respuesta,
            "revisor": revisor,
            "job_id": job_id,
            "image_ids": list(image_ids) if image_ids else [],
        }
        with self._lock:
            self._entries.append(entry)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._entries)

    def to_cases(self) -> list[Case]:
        """Convertir los casos aprobados en `Case` para sumarlos a la base buscable."""
        with self._lock:
            entries = list(self._entries)
        return [
            Case(
                encounter_id=e["encounter_id"],
                split=SPLIT_APROBADO,
                query_title=e.get("consulta", ""),
                query_content="",
                query_text=e.get("consulta", ""),
                answer=e.get("respuesta", ""),
                image_ids=e.get("image_ids", []) or [],
            )
            for e in entries
        ]
