"""
Tests del upgrade de seguridad (Fase 5), construidos a partir de los casos reales
que los dermatólogos de prueba lograron colar como "nivel bajo":
- banderas rojas sobre la CONSULTA (fuerzan nivel alto),
- falsa tranquilización (descartar malignidad con banderas presentes),
- evidencia débil (similitud por debajo del umbral).
"""

from app.safety.analyzer import (
    analizar,
    banderas_rojas,
    falsa_tranquilizacion,
)

# ── Banderas rojas sobre la consulta ─────────────────────────────────────────

def test_bandera_melanoma_pigmentada():
    assert "sospecha_melanoma" in banderas_rojas("Sospecha de melanoma en la espalda")
    b = banderas_rojas("Un lunar que cambió de color y ahora sangra")
    assert "lesion_pigmentada_cambiante" in b and "sangrado" in b


def test_bandera_lesion_acral():
    assert "lesion_acral" in banderas_rojas("Lesión nueva asimétrica de 8 mm en la planta del pie")


def test_bandera_ulcera_no_cicatriza():
    assert "lesion_no_cicatriza" in banderas_rojas("Úlcera que no cierra hace 4 meses en el pie")


def test_bandera_urgencia_sistemica():
    b = banderas_rojas("Fiebre alta con ampollas y se despega la piel")
    assert "urgencia_sistemica" in b


def test_consulta_benigna_sin_banderas():
    assert banderas_rojas("Control de psoriasis conocida en placas estables") == []


# ── Integración: las banderas fuerzan nivel alto ─────────────────────────────

def test_banderas_fuerzan_nivel_alto():
    # Aunque el borrador copiado sea inocuo y coincida con la evidencia.
    evidencia = ["El cuadro es compatible con nevo; controlar."]
    borrador = "El cuadro es compatible con nevo; se recomienda control."
    res = analizar(
        borrador, evidencia,
        consulta="Lunar que creció y sangra en el último mes", similitud_max=0.6,
    )
    assert res["banderas_rojas"]
    assert res["nivel"] == "alto"


def test_falsa_tranquilizacion_escala_a_alto():
    # El peor caso: descartar un melanoma que la consulta hace sospechar.
    res = analizar(
        "Por la clínica se puede descartar un melanoma; no es preocupante.",
        ["Lesión pigmentada; controlar."],
        consulta="Mancha que creció, cambió de color y sangra; ¿melanoma?",
        similitud_max=0.5,
    )
    assert res["falsa_tranquilizacion"] is True
    assert res["nivel"] == "alto"


def test_falsa_tranquilizacion_no_dispara_sin_banderas():
    assert falsa_tranquilizacion("Es benigno, no es preocupante.", hay_banderas=False) is False


# ── Confianza: evidencia débil ───────────────────────────────────────────────

def test_evidencia_debil_eleva_a_medio():
    res = analizar(
        "El cuadro es compatible con eccema; hidratar la piel.",
        ["El cuadro es compatible con eccema; hidratar."],
        consulta="Piel seca en los brazos", similitud_max=0.20,
    )
    assert res["evidencia_debil"] is True
    assert res["similitud_max"] == 0.20
    assert res["nivel"] in ("medio", "alto")


def test_evidencia_fuerte_no_marca_debil():
    borrador = (
        "El cuadro es compatible con eccema; se recomienda hidratar la piel "
        "con emolientes y evitar el jabon fuerte."
    )
    res = analizar(
        borrador,
        ["El cuadro es compatible con eccema; hidratar."],
        consulta="Piel seca en los brazos", similitud_max=0.7,
    )
    assert res["evidencia_debil"] is False
    assert res["nivel"] == "bajo"


# ── Compatibilidad hacia atrás ───────────────────────────────────────────────

def test_analizar_sin_consulta_ni_similitud():
    # La firma vieja (solo borrador + evidencia) sigue funcionando.
    res = analizar(
        "El cuadro es compatible con eccema; se recomienda hidratar la piel y evitar irritantes locales.",
        ["compatible con eccema"],
    )
    assert res["banderas_rojas"] == []
    assert res["evidencia_debil"] is False
    assert res["similitud_max"] is None
    assert res["nivel"] == "bajo"
