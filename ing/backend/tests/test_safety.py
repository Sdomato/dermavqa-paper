"""
Tests de la capa de seguridad (Fase 3). Lógica pura, sin modelos.
"""

from app.safety.analyzer import (
    analizar,
    condiciones_en,
    diagnosticos_no_sustentados,
    flags_heuristicos,
    terminos_riesgo,
)

# ── heurísticos ─────────────────────────────────────────────────────────────

def test_flag_vacio():
    assert flags_heuristicos("") == ["vacio"]
    assert flags_heuristicos("   ") == ["vacio"]


def test_flag_muy_corto():
    assert "muy_corto" in flags_heuristicos("Es psoriasis.")


def test_flag_repetitivo():
    # Degeneración tipo ENC00923: misma oración repetida.
    txt = ("La tina de manos es causada por hongos. " * 4)
    assert "repetitivo" in flags_heuristicos(txt)


def test_sin_flags_texto_normal():
    txt = ("El cuadro es compatible con dermatitis de contacto; se recomienda "
           "evitar el alergeno y usar emolientes durante dos semanas.")
    assert flags_heuristicos(txt) == []


# ── condiciones / grounding ─────────────────────────────────────────────────

def test_condiciones_detecta_con_y_sin_acento():
    assert "acne" in condiciones_en("Tiene acné en la cara")
    assert "psoriasis" in condiciones_en("Compatible con PSORIASIS")


def test_grounding_marca_diagnostico_no_en_evidencia():
    # Borrador dice psoriasis; la evidencia hablaba de liquen plano.
    no_sus = diagnosticos_no_sustentados(
        "El cuadro es compatible con psoriasis.",
        ["El cuadro es compatible con liquen plano."],
    )
    assert "psoriasis" in no_sus
    assert "liquen plano" not in no_sus


def test_grounding_ok_cuando_coincide():
    no_sus = diagnosticos_no_sustentados(
        "Compatible con eccema.", ["Probable eccema dishidrótico."]
    )
    assert no_sus == []


# ── términos de riesgo ──────────────────────────────────────────────────────

def test_riesgo_detecta_procedimiento_y_farmaco():
    r = terminos_riesgo("Se sugiere biopsia y tratamiento con antibióticos.")
    cats = {x["categoria"] for x in r}
    assert "procedimiento_invasivo" in cats and "farmaco_sistemico" in cats


# ── nivel global ────────────────────────────────────────────────────────────

def test_nivel_alto_por_diagnostico_no_sustentado():
    out = analizar("Es psoriasis.", ["Es liquen plano."])
    assert out["nivel"] == "alto"
    assert "psoriasis" in out["diagnosticos_no_sustentados"]


def test_nivel_alto_por_riesgo():
    out = analizar(
        "Compatible con eccema; se sugiere biopsia para confirmar el diagnóstico.",
        ["Compatible con eccema, considerar biopsia."],
    )
    assert out["nivel"] == "alto"  # biopsia (procedimiento invasivo)


def test_nivel_medio_por_flag_sin_riesgo_ni_alucinacion():
    # Corto pero fundado y sin recomendación riesgosa → medio.
    out = analizar("Es dermatitis.", ["Compatible con dermatitis."])
    assert out["nivel"] == "medio"
    assert "muy_corto" in out["flags"]
    assert out["diagnosticos_no_sustentados"] == []


def test_nivel_bajo_texto_seguro_y_fundado():
    out = analizar(
        "El cuadro es compatible con dermatitis; se recomienda hidratación y evitar irritantes.",
        ["Compatible con dermatitis de contacto; hidratar y evitar irritantes."],
    )
    assert out["nivel"] == "bajo"
    assert out["diagnosticos_no_sustentados"] == [] and out["flags"] == []


# ── integración vía /borrador ───────────────────────────────────────────────

def test_borrador_incluye_seguridad(client):
    import time
    r = client.post("/borrador", data={"titulo": "lesion que pica", "k": "3"})
    job_id = r.json()["job_id"]
    for _ in range(40):
        body = client.get(f"/borrador/{job_id}").json()
        if body["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert body["status"] == "done"
    seg = body["seguridad"]
    assert seg is not None
    assert seg["nivel"] in ("bajo", "medio", "alto")
    assert "flags" in seg and "diagnosticos_no_sustentados" in seg
