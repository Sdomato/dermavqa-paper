"""
Tests de los dos modos de falla del paper en la capa de seguridad:
cambio de entidad diagnóstica y recomendaciones no sustentadas.
Los ejemplos están inspirados en casos reales de la revisión clínica (ENC00909, ENC00935).
"""

from app.safety.analyzer import (
    analizar,
    cambio_de_entidad,
    recomendaciones_no_sustentadas,
)

# ── Cambio de entidad diagnóstica ────────────────────────────────────────────

def test_cambio_de_entidad_detecta_desplazamiento():
    # ENC00909: referencia linfangioma/nevus → borrador psoriasis.
    evidencia = ["El cuadro es compatible con linfangioma, nevus epidérmico lineal o penfigoide."]
    borrador = "El cuadro es compatible con psoriasis; se sugiere biopsia."
    assert cambio_de_entidad(borrador, evidencia) is True


def test_no_hay_cambio_si_comparten_diagnostico():
    evidencia = ["El cuadro es compatible con psoriasis, a confirmar con biopsia."]
    borrador = "El cuadro es compatible con psoriasis; también dermatitis de contacto."
    assert cambio_de_entidad(borrador, evidencia) is False


def test_sin_diagnosticos_no_marca_cambio():
    assert cambio_de_entidad("Se recomienda hidratación y control.", ["Aplicar crema."]) is False


# ── Recomendaciones no sustentadas ───────────────────────────────────────────

def test_recomendacion_no_sustentada():
    # ENC00935: referencia liquen plano (sin estudios) → borrador con análisis de sangre.
    evidencia = ["El cuadro es compatible con liquen plano, pápulas planas de color púrpura."]
    borrador = (
        "El cuadro es compatible con erupción por medicamentos. "
        "Se sugiere realizar pruebas de sangre y considerar antibióticos."
    )
    recs = recomendaciones_no_sustentadas(borrador, evidencia)
    assert "pruebas de sangre" in recs
    assert "antibioticos" in recs


def test_recomendacion_sustentada_no_se_marca():
    evidencia = ["Se recomienda una prueba de hongos para confirmar la infección."]
    borrador = "Sugiero una prueba de hongos para confirmar."
    assert recomendaciones_no_sustentadas(borrador, evidencia) == []


# ── Integración en el nivel global ───────────────────────────────────────────

def test_cambio_de_entidad_eleva_a_alto():
    evidencia = ["El cuadro es compatible con angioma senil."]
    borrador = "El cuadro es compatible con granuloma anular; se sugiere seguimiento clínico."
    res = analizar(borrador, evidencia)
    assert res["cambio_de_entidad"] is True
    assert res["nivel"] == "alto"


def test_recomendacion_no_sustentada_eleva_a_medio():
    # Sin cambio de entidad ni diagnóstico no sustentado, pero recomienda un estudio ausente.
    evidencia = ["El cuadro es compatible con eccema; mantener la piel hidratada."]
    borrador = "El cuadro es compatible con eccema; conviene una dermatoscopia de control."
    res = analizar(borrador, evidencia)
    assert "dermatoscopia" in res["recomendaciones_no_sustentadas"]
    assert res["cambio_de_entidad"] is False
    assert res["nivel"] in ("medio", "alto")


def test_borrador_alineado_queda_bajo():
    evidencia = ["El cuadro es compatible con dermatitis de contacto; evitar el alérgeno."]
    borrador = (
        "El cuadro es compatible con dermatitis de contacto; "
        "se recomienda evitar el alérgeno y usar emolientes."
    )
    res = analizar(borrador, evidencia)
    assert res["cambio_de_entidad"] is False
    assert res["recomendaciones_no_sustentadas"] == []
    assert res["nivel"] == "bajo"
