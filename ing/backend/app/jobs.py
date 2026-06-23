"""
Cola de trabajos en memoria.

La generación con el VLM tarda 12–26 s, así que el endpoint no puede bloquear:
encola el trabajo, devuelve un job_id, y el cliente hace poll del estado. Acá
usamos un ThreadPoolExecutor en proceso (simple y sin infra extra). En producción
esto se reemplazaría por una cola real (Redis/Celery) con persistencia.
"""

import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any


@dataclass
class Job:
    id: str
    status: str = "pending"  # pending | running | done | error
    result: Any = None
    error: str | None = None


class JobStore:
    def __init__(self, max_workers: int = 1) -> None:
        # 1 worker por defecto: el VLM es pesado, conviene serializar.
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, fn: Callable[[], Any]) -> str:
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id)
        with self._lock:
            self._jobs[job_id] = job

        def _run() -> None:
            with self._lock:
                job.status = "running"
            try:
                result = fn()
                with self._lock:
                    job.result = result
                    job.status = "done"
            except Exception as exc:  # noqa: BLE001 — queremos capturar todo y reportarlo
                with self._lock:
                    job.error = str(exc)
                    job.status = "error"

        self._executor.submit(_run)
        return job_id

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)
