"""
Tests de la generación de borradores (Fase 2).

Con el generador `stub` por defecto todo el flujo (prompt RAG, cola async,
endpoints, poll) se verifica sin GPU ni VLM.
"""

import time

from app.generation.prompt import build_rag_prompt
from app.generation.stub import StubGenerator

# ── prompt RAG (puro) ───────────────────────────────────────────────────────

def test_prompt_incluye_consulta_y_evidencia():
    evidence = [{"answer": "Compatible con psoriasis.", "similitud": 0.8}]
    p = build_rag_prompt("Mancha roja que pica", evidence)
    assert "Mancha roja que pica" in p
    assert "psoriasis" in p
    assert "inventes" in p.lower()  # instrucción anti-alucinación presente


def test_prompt_sin_evidencia_lo_dice():
    p = build_rag_prompt("consulta", [])
    assert "No se encontraron casos" in p


# ── stub generator ──────────────────────────────────────────────────────────

def test_stub_usa_evidencia():
    gen = StubGenerator()
    out = gen.generate("consulta", [{"answer": "Tratar con corticoides."}])
    assert "corticoides" in out
    assert "revisión" in out.lower()


def test_stub_sin_evidencia():
    out = StubGenerator().generate("consulta", [])
    assert "No se encontraron" in out


# ── flujo async vía endpoints ───────────────────────────────────────────────

def _poll(client, job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/borrador/{job_id}")
        assert r.status_code == 200
        body = r.json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.05)
    raise AssertionError("el job no terminó a tiempo")


def test_borrador_flujo_completo(client):
    r = client.post("/borrador", data={"titulo": "lesión que pica", "k": "3"})
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    final = _poll(client, job_id)
    assert final["status"] == "done"
    assert final["borrador"]  # hay texto
    assert final["evidencia"] and len(final["evidencia"]) <= 3
    # El stub fundamenta en el caso más similar.
    assert "revisión" in final["borrador"].lower()


def test_borrador_con_imagen(client):
    # Ejercita el guardado de upload + limpieza de temporales en el task async.
    png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00"
        b"\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    r = client.post(
        "/borrador",
        data={"titulo": "lesión pigmentada", "k": "2"},
        files={"imagenes": ("lesion.png", png, "image/png")},
    )
    assert r.status_code == 200
    final = _poll(client, r.json()["job_id"])
    assert final["status"] == "done"
    assert final["borrador"]


def test_borrador_vacio_422(client):
    assert client.post("/borrador", data={"titulo": "", "contenido": ""}).status_code == 422


def test_borrador_job_inexistente_404(client):
    assert client.get("/borrador/noexiste123").status_code == 404
