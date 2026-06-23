"""
Tests del flujo de revisión médica + audit log (Fase 3).
"""

import time

import pytest

from app import main
from app.audit import AuditLog

# ── AuditLog (unidad, con archivo temporal) ─────────────────────────────────

def test_auditlog_append_y_list(tmp_path):
    log = AuditLog(tmp_path / "rev.jsonl")
    e = log.append({"job_id": "j1", "accion": "aprobar"})
    assert e["id"] and e["timestamp"]
    assert len(log.list()) == 1 and log.list()[0]["accion"] == "aprobar"


def test_auditlog_persiste(tmp_path):
    p = tmp_path / "rev.jsonl"
    AuditLog(p).append({"job_id": "j1", "accion": "rechazar"})
    # Una instancia nueva debe leer lo que quedó en disco.
    otra = AuditLog(p)
    assert len(otra.list()) == 1 and otra.list()[0]["accion"] == "rechazar"


# ── Flujo vía endpoints ─────────────────────────────────────────────────────

@pytest.fixture
def audit_temporal(tmp_path, monkeypatch):
    """Aísla el audit log del servicio en un archivo temporal por test."""
    monkeypatch.setattr(main, "_audit", AuditLog(tmp_path / "rev.jsonl"))


def _borrador_listo(client):
    job_id = client.post("/borrador", data={"titulo": "lesion que pica", "k": "3"}).json()["job_id"]
    for _ in range(40):
        if client.get(f"/borrador/{job_id}").json()["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    return job_id


def test_aprobar_y_aparece_en_auditoria(client, audit_temporal):
    job_id = _borrador_listo(client)
    r = client.post(f"/borrador/{job_id}/revision", json={"accion": "aprobar", "revisor": "dra. lopez"})
    assert r.status_code == 200
    entry = r.json()
    assert entry["accion"] == "aprobar" and entry["texto_final"]  # aprobar conserva el borrador
    assert entry["seguridad_nivel"] in ("bajo", "medio", "alto")

    aud = client.get("/auditoria").json()
    assert aud["total"] == 1 and aud["revisiones"][0]["job_id"] == job_id


def test_editar_guarda_texto_final(client, audit_temporal):
    job_id = _borrador_listo(client)
    r = client.post(f"/borrador/{job_id}/revision",
                    json={"accion": "editar", "texto_final": "Texto corregido por el médico."})
    assert r.status_code == 200
    assert r.json()["texto_final"] == "Texto corregido por el médico."


def test_rechazar_sin_texto(client, audit_temporal):
    job_id = _borrador_listo(client)
    r = client.post(f"/borrador/{job_id}/revision", json={"accion": "rechazar", "nota": "no aplica"})
    assert r.status_code == 200 and r.json()["texto_final"] is None


def test_accion_invalida_422(client, audit_temporal):
    job_id = _borrador_listo(client)
    assert client.post(f"/borrador/{job_id}/revision", json={"accion": "borrar"}).status_code == 422


def test_editar_sin_texto_422(client, audit_temporal):
    job_id = _borrador_listo(client)
    assert client.post(f"/borrador/{job_id}/revision", json={"accion": "editar"}).status_code == 422


def test_revision_job_inexistente_404(client, audit_temporal):
    assert client.post("/borrador/nope/revision", json={"accion": "aprobar"}).status_code == 404
