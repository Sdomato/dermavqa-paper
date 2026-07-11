"""
Tests de integridad detectados por el testeo QA:
- un borrador se revisa una sola vez (idempotencia).
- /borrador valida k >= 1 igual que /consulta.
"""

import time

import pytest

from app import main
from app.audit import AuditLog
from app.feedback import CasosAprobadosStore


@pytest.fixture
def estado_temporal(tmp_path, client):
    """Aísla audit + aprobados y restaura el índice al terminar."""
    orig_audit, orig_apr = main._audit, main._aprobados
    main._audit = AuditLog(tmp_path / "rev.jsonl")
    main._aprobados = CasosAprobadosStore(tmp_path / "apr.jsonl")
    main._rebuild_index()
    try:
        yield
    finally:
        main._audit, main._aprobados = orig_audit, orig_apr
        main._rebuild_index()


def _borrador_listo(client, titulo="lesion que pica"):
    job_id = client.post("/borrador", data={"titulo": titulo, "k": "3"}).json()["job_id"]
    for _ in range(40):
        if client.get(f"/borrador/{job_id}").json()["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    return job_id


# ── Fix #1: idempotencia de la revisión ──────────────────────────────────────

def test_doble_revision_devuelve_409(client, estado_temporal):
    job_id = _borrador_listo(client)
    r1 = client.post(f"/borrador/{job_id}/revision", json={"accion": "aprobar", "revisor": "dra. paz"})
    assert r1.status_code == 200
    # Segunda revisión del mismo job → rechazada.
    r2 = client.post(f"/borrador/{job_id}/revision", json={"accion": "rechazar"})
    assert r2.status_code == 409


def test_revision_no_duplica_caso_aprobado(client, estado_temporal):
    job_id = _borrador_listo(client, "prurito localizado unico")
    client.post(f"/borrador/{job_id}/revision", json={"accion": "aprobar"})
    # Reintento de aprobación: no debe agregar otro caso.
    client.post(f"/borrador/{job_id}/revision", json={"accion": "aprobar"})
    aprobados = client.get("/dataset/aprobados").json()
    del_job = [c for c in aprobados["casos"] if c["job_id"] == job_id]
    assert len(del_job) == 1


# ── Fix #2: validación de k en /borrador ─────────────────────────────────────

def test_borrador_k_cero_422(client):
    assert client.post("/borrador", data={"titulo": "algo", "k": "0"}).status_code == 422


def test_borrador_k_negativo_422(client):
    assert client.post("/borrador", data={"titulo": "algo", "k": "-5"}).status_code == 422


def test_consulta_imagen_k_invalido_422(client):
    assert client.post("/consulta/imagen", data={"titulo": "algo", "k": "0"}).status_code == 422
