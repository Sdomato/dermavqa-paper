"""
Harness de evaluación del servicio (calidad de los borradores en producción).

Trae al servicio la misma idea de las métricas del paper, pero aplicada al dato
que genera el uso real: cada revisión médica compara el **borrador automático**
con el **texto final aprobado**. Cuanto menos tiene que editar el médico, mejor
era el borrador. Sumado a la distribución de niveles de seguridad, da una señal
concreta y sin GPU de si el sistema mejora con el tiempo (loop de la Fase 4).

Métricas ligeras y puras (sin deps pesadas):
- token-F1 entre borrador y texto final → cuánto se conservó (proxy de calidad).
- edición = 1 - token-F1 → cuánto tuvo que reescribir el médico.

No mide corrección clínica (eso lo hace la revisión humana); mide cuán útil fue
el borrador como punto de partida.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any


def _tokens(texto: str) -> list[str]:
    t = unicodedata.normalize("NFKD", str(texto or "").lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.findall(r"\w+", t)


def token_f1(a: str, b: str) -> float:
    """F1 de solapamiento de tokens (multiconjunto) entre dos textos, en [0, 1]."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    solapan = sum((Counter(ta) & Counter(tb)).values())
    if solapan == 0:
        return 0.0
    precision = solapan / len(ta)
    recall = solapan / len(tb)
    return 2 * precision * recall / (precision + recall)


def _promedio(valores: list[float]) -> float | None:
    return round(sum(valores) / len(valores), 4) if valores else None


def resumen_auditoria(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Agrega las revisiones en indicadores de calidad del servicio.

    - por_accion / tasa_aprobacion: cuánto se acepta vs. se rechaza.
    - similitud_borrador_final / edicion_media: cuánto sobrevive el borrador (calidad).
    - por_nivel_seguridad: distribución de alertas (debería tender a 'bajo').
    """
    total = len(entries)
    por_accion = Counter(e.get("accion", "") for e in entries)
    por_nivel = Counter(e.get("seguridad_nivel") or "desconocido" for e in entries)

    # Similitud borrador→final solo donde hay ambos textos (aprobar/editar).
    sims: list[float] = []
    for e in entries:
        original = e.get("borrador_original")
        final = e.get("texto_final")
        if original and final:
            sims.append(token_f1(original, final))

    aceptadas = por_accion.get("aprobar", 0) + por_accion.get("editar", 0)
    similitud = _promedio(sims)

    return {
        "total_revisiones": total,
        "por_accion": dict(por_accion),
        "tasa_aprobacion": round(aceptadas / total, 4) if total else None,
        "similitud_borrador_final": similitud,
        "edicion_media": round(1 - similitud, 4) if similitud is not None else None,
        "por_nivel_seguridad": dict(por_nivel),
    }
