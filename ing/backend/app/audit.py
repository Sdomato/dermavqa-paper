"""
Registro de auditoría (audit log) de las revisiones médicas.

Cada vez que un médico aprueba / edita / rechaza un borrador, queda una entrada
append-only. Se persiste en un archivo JSONL (una entrada por línea) para que
sobreviva reinicios y —lo más importante— para que el conjunto de revisiones se
vuelva el **dataset de validación clínica humana** que al paper le falta.

En producción esto iría a una base con trazabilidad; acá un JSONL alcanza para
el MVP y es inspeccionable a mano.
"""

import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class AuditLog:
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

    def append(self, entry: dict[str, Any]) -> dict[str, Any]:
        e = dict(entry)
        e["id"] = uuid.uuid4().hex[:12]
        e["timestamp"] = datetime.now(UTC).isoformat()
        with self._lock:
            self._entries.append(e)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        return e

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._entries)
