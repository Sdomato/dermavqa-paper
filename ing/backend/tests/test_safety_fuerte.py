"""
Test fuerte de la capa de seguridad (Fase 3).

Dos frentes:
1. Adversariales / casos borde de la lógica (límites de palabra, acentos, crashes).
2. Validación contra los 20 casos REALES del clinical review del paper: ¿la capa
   marca como riesgosos los borradores que el review humano juzgó incorrectos o
   con alucinaciones? Se exige un recall alto sobre esos casos problemáticos.
"""

import csv
from pathlib import Path

import pytest

from app.safety.analyzer import (
    analizar,
    condiciones_en,
    diagnosticos_no_sustentados,
    flags_heuristicos,
    terminos_riesgo,
)

_REVIEW_CSV = Path(__file__).resolve().parents[3] / "outputs/paper/tables/paper_clinical_review_20.csv"


# ── adversariales / borde ────────────────────────────────────────────────────

def test_limite_de_palabra_no_falso_match():
    # "tina" (tiña) no debe matchear dentro de "cortina"; "nevo" no en "nuevo".
    assert "tina" not in condiciones_en("La cortina del baño está sucia")
    assert "nevo" not in condiciones_en("Compré algo nuevo ayer")


def test_acentos_y_mayusculas():
    assert "tina" in condiciones_en("Tiene TIÑA en el pie")
    assert "acne" in condiciones_en("Acné quístico")
    assert "psoriasis" in condiciones_en("PSORIASIS en placas")


def test_evidencia_vacia_marca_todo():
    # Sin evidencia, cualquier diagnóstico del borrador queda no sustentado.
    out = diagnosticos_no_sustentados("Compatible con psoriasis y eccema.", [])
    assert set(out) == {"psoriasis", "eccema"}


def test_dedup_y_multiples_evidencias():
    no_sus = diagnosticos_no_sustentados(
        "Psoriasis, psoriasis y liquen plano.",
        ["Habla de liquen plano.", "Otro caso de dermatitis."],
    )
    assert no_sus == ["psoriasis"]  # liquen plano está en evidencia; psoriasis no, sin duplicar


def test_riesgo_con_acentos():
    r = {x["termino"] for x in terminos_riesgo("Indicar antibióticos y corticoides; valorar biopsia.")}
    assert {"antibioticos", "corticoides", "biopsia"} <= r


def test_repetitivo_umbral():
    base = "El cuadro es compatible con dermatitis"
    assert "repetitivo" not in flags_heuristicos(f"{base}. {base}.")          # 2 veces: no
    assert "repetitivo" in flags_heuristicos(f"{base}. {base}. {base}.")      # 3 veces: sí


def test_no_crashea_con_entrada_rara():
    for raro in ["", "   ", "💊🔬", "a" * 5000, "...,,,;;;", "123 456"]:
        out = analizar(raro, [])
        assert out["nivel"] in ("bajo", "medio", "alto")


# ── validación contra datos reales del paper ─────────────────────────────────

@pytest.mark.skipif(not _REVIEW_CSV.exists(), reason="falta paper_clinical_review_20.csv")
def test_validacion_contra_clinical_review():
    rows = list(csv.DictReader(_REVIEW_CSV.open(encoding="utf-8")))
    assert len(rows) >= 15

    problematicos = detectados = 0
    for r in rows:
        out = analizar(r["predicted_answer_es"], [r["reference_answer_es"]])
        es_problematico = (
            r["clinical_correctness"] == "incorrect"
            or r["hallucination_or_invented_info"] == "yes"
            or r["recommendation_safety"] in ("unsupported", "potentially_unsafe")
        )
        marcado = out["nivel"] in ("medio", "alto") or bool(out["diagnosticos_no_sustentados"])
        if es_problematico:
            problematicos += 1
            detectados += int(marcado)

    recall = detectados / problematicos
    # La capa debe atrapar la gran mayoría de los borradores problemáticos reales.
    assert recall >= 0.85, f"recall insuficiente: {detectados}/{problematicos}"


@pytest.mark.skipif(not _REVIEW_CSV.exists(), reason="falta paper_clinical_review_20.csv")
def test_casos_de_cambio_de_diagnostico_marcados():
    # Casos donde el VLM cambió la entidad diagnóstica (ej. ENC00908/ENC00909):
    # deben quedar en nivel alto con diagnósticos no sustentados.
    rows = {r["encounter_id"]: r for r in csv.DictReader(_REVIEW_CSV.open(encoding="utf-8"))}
    for enc in ("ENC00908", "ENC00909"):
        if enc not in rows:
            continue
        out = analizar(rows[enc]["predicted_answer_es"], [rows[enc]["reference_answer_es"]])
        assert out["nivel"] == "alto"
        assert out["diagnosticos_no_sustentados"]
