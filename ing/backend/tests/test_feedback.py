"""
Tests del loop de mejora (Fase 4): casos aprobados que retroalimentan la base.
"""

import time

import pytest

from app import main
from app.audit import AuditLog
from app.feedback import SPLIT_APROBADO, CasosAprobadosStore
from scripts.build_finetune_dataset import construir

# ── CasosAprobadosStore (unidad, con archivo temporal) ──────────────────────

def test_store_agregar_y_list(tmp_path):
    store = CasosAprobadosStore(tmp_path / "apr.jsonl")
    e = store.agregar(consulta="lesion que pica", respuesta="Aplicar crema.", revisor="dra. lopez")
    assert e["encounter_id"].startswith("APR-") and e["timestamp"]
    assert len(store.list()) == 1 and store.list()[0]["respuesta"] == "Aplicar crema."


def test_store_persiste(tmp_path):
    p = tmp_path / "apr.jsonl"
    CasosAprobadosStore(p).agregar(consulta="q", respuesta="r")
    # Una instancia nueva lee lo que quedó en disco.
    otra = CasosAprobadosStore(p)
    assert len(otra.list()) == 1 and otra.list()[0]["consulta"] == "q"


def test_store_to_cases(tmp_path):
    store = CasosAprobadosStore(tmp_path / "apr.jsonl")
    store.agregar(consulta="mancha oscura en la espalda", respuesta="Derivar a control.")
    cases = store.to_cases()
    assert len(cases) == 1
    c = cases[0]
    assert c.split == SPLIT_APROBADO
    assert c.query_text == "mancha oscura en la espalda"
    assert c.answer == "Derivar a control."
    assert c.encounter_id.startswith("APR-")


# ── Export para retrain (unidad) ─────────────────────────────────────────────

def test_construir_dataset_retrain(tmp_path):
    apr = tmp_path / "apr.jsonl"
    store = CasosAprobadosStore(apr)
    # Un caso con imagen (entra al retrain del LoRA) y otro solo texto.
    store.agregar(consulta="caso con foto", respuesta="resp 1", image_ids=["IMG1"])
    store.agregar(consulta="caso solo texto", respuesta="resp 2")

    salida = tmp_path / "out.jsonl"
    stats = construir(apr, salida)
    assert stats["total"] == 2
    assert stats["con_imagen_para_lora"] == 1
    assert stats["solo_texto"] == 1
    assert salida.exists() and len(salida.read_text().strip().splitlines()) == 2


# ── Flujo end-to-end vía endpoints ───────────────────────────────────────────

@pytest.fixture
def loop_temporal(tmp_path, client):
    """Aísla el store de aprobados y el audit log, y restaura el índice al terminar."""
    orig_apr, orig_audit = main._aprobados, main._audit
    main._aprobados = CasosAprobadosStore(tmp_path / "apr.jsonl")
    main._audit = AuditLog(tmp_path / "rev.jsonl")
    main._rebuild_index()
    try:
        yield main._aprobados
    finally:
        main._aprobados, main._audit = orig_apr, orig_audit
        main._rebuild_index()


def _borrador_listo(client, titulo):
    job_id = client.post("/borrador", data={"titulo": titulo, "k": "3"}).json()["job_id"]
    for _ in range(40):
        if client.get(f"/borrador/{job_id}").json()["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    return job_id


def test_aprobar_promueve_y_es_recuperable(client, loop_temporal):
    # Consulta con tokens raros para que el caso aprobado domine el retrieval.
    consulta = "xanto fucsia triangular zorzal"
    indexados_antes = client.get("/health").json()["casos_indexados"]

    job_id = _borrador_listo(client, consulta)
    r = client.post(f"/borrador/{job_id}/revision", json={"accion": "aprobar", "revisor": "dr. paz"})
    assert r.status_code == 200

    # Definition of done #1: aparece en el dataset humano-validado.
    ds = client.get("/dataset/aprobados").json()
    assert ds["total"] == 1
    aprobado = ds["casos"][0]
    assert aprobado["consulta"] == consulta and aprobado["encounter_id"].startswith("APR-")

    # Definition of done #2: quedó indexado (un caso más) y es recuperable.
    salud = client.get("/health").json()
    assert salud["casos_indexados"] == indexados_antes + 1
    assert salud["casos_aprobados"] == 1

    hits = client.post("/consulta", json={"titulo": consulta}).json()["resultados"]
    ids = [h["encounter_id"] for h in hits]
    assert aprobado["encounter_id"] in ids, "el caso aprobado debería recuperarse para su propia consulta"


def test_editar_promueve_texto_corregido(client, loop_temporal):
    job_id = _borrador_listo(client, "prurito nocturno persistente kappa")
    r = client.post(
        f"/borrador/{job_id}/revision",
        json={"accion": "editar", "texto_final": "Indico antihistamínico y control en 7 días."},
    )
    assert r.status_code == 200
    casos = client.get("/dataset/aprobados").json()["casos"]
    assert casos[0]["respuesta"] == "Indico antihistamínico y control en 7 días."


def test_rechazar_no_promueve(client, loop_temporal):
    job_id = _borrador_listo(client, "consulta cualquiera omicron")
    r = client.post(f"/borrador/{job_id}/revision", json={"accion": "rechazar", "nota": "no aplica"})
    assert r.status_code == 200
    assert client.get("/dataset/aprobados").json()["total"] == 0
