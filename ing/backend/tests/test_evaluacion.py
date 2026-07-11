"""
Tests del harness de evaluación: métricas ligeras y resumen del audit log.
"""

from app.evaluacion import resumen_auditoria, token_f1

# ── token_f1 ─────────────────────────────────────────────────────────────────

def test_token_f1_identico():
    assert token_f1("aplicar crema dos veces al dia", "aplicar crema dos veces al dia") == 1.0


def test_token_f1_disjunto():
    assert token_f1("psoriasis", "onicomicosis foliculitis") == 0.0


def test_token_f1_parcial():
    v = token_f1("aplicar crema con corticoides", "aplicar crema por la manana")
    assert 0.0 < v < 1.0


def test_token_f1_vacios():
    assert token_f1("", "") == 1.0
    assert token_f1("algo", "") == 0.0


# ── resumen_auditoria ────────────────────────────────────────────────────────

def _entry(accion, original=None, final=None, nivel="bajo"):
    return {
        "accion": accion,
        "borrador_original": original,
        "texto_final": final,
        "seguridad_nivel": nivel,
    }


def test_resumen_vacio():
    r = resumen_auditoria([])
    assert r["total_revisiones"] == 0
    assert r["tasa_aprobacion"] is None
    assert r["similitud_borrador_final"] is None


def test_resumen_cuenta_acciones_y_niveles():
    entries = [
        _entry("aprobar", "texto base del borrador", "texto base del borrador", "bajo"),
        _entry("editar", "texto base del borrador", "texto totalmente distinto reescrito", "alto"),
        _entry("rechazar", nivel="alto"),
    ]
    r = resumen_auditoria(entries)
    assert r["total_revisiones"] == 3
    assert r["por_accion"] == {"aprobar": 1, "editar": 1, "rechazar": 1}
    # 2 de 3 aceptadas (aprobar + editar).
    assert r["tasa_aprobacion"] == round(2 / 3, 4)
    assert r["por_nivel_seguridad"] == {"bajo": 1, "alto": 2}


def test_similitud_refleja_edicion():
    # Aprobado sin cambios → similitud alta; muy editado → baja. El promedio queda en el medio.
    entries = [
        _entry("aprobar", "un borrador clinico conciso", "un borrador clinico conciso"),
        _entry("editar", "un borrador clinico conciso", "otra cosa completamente diferente aqui"),
    ]
    r = resumen_auditoria(entries)
    assert r["similitud_borrador_final"] is not None
    assert 0.0 < r["similitud_borrador_final"] < 1.0
    # edicion_media es el complemento de la similitud.
    assert abs(r["edicion_media"] + r["similitud_borrador_final"] - 1.0) < 1e-6
