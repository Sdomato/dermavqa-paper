"""
Análisis de seguridad de un borrador (Fase 3).

Todo es lógica pura y determinística (testeable sin modelos):

1. Heurísticos      → vacío / muy corto / repetitivo.
2. Grounding        → diagnósticos nombrados en el borrador que NO aparecen en la
                      evidencia recuperada.
3. Cambio de entidad → el diagnóstico del borrador no coincide con el del caso más
                      parecido (primer modo de falla del paper: el modelo desplaza
                      la entidad diagnóstica central, ej. linfangioma → psoriasis).
4. Recomendaciones no sustentadas → estudios/tratamientos sugeridos por el borrador
                      que NO aparecen en la evidencia (segundo modo de falla del
                      paper: propone biopsias, análisis o tratamientos ausentes en
                      la referencia).
5. Términos de riesgo → recomendaciones sensibles (biopsia, antibióticos, etc.).
6. Nivel global     → low / medium / high, para priorizar la revisión médica.

El objetivo es SEÑALAR para el revisor humano, no decidir clínicamente.
"""

import re
import unicodedata
from typing import Any

from .lexicon import CONDICIONES, RECOMENDACIONES, TERMINOS_RIESGO

MIN_PALABRAS = 12


def _norm(texto: str) -> str:
    """Minúsculas y sin acentos, para matchear de forma robusta."""
    t = unicodedata.normalize("NFKD", str(texto or "").lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def _contiene_palabra(termino: str, texto_norm: str) -> bool:
    """Match por límite de palabra (evita que 'acne' matchee dentro de otra)."""
    return re.search(rf"\b{re.escape(termino)}\b", texto_norm) is not None


def flags_heuristicos(borrador: str, min_palabras: int = MIN_PALABRAS) -> list[str]:
    flags: list[str] = []
    t = (borrador or "").strip()
    if not t:
        return ["vacio"]
    if len(t.split()) < min_palabras:
        flags.append("muy_corto")
    # Repetitivo: alguna oración aparece 3+ veces (degeneración del modelo).
    oraciones = [s.strip().lower() for s in re.split(r"[.\n]", t) if len(s.strip()) > 8]
    for o in set(oraciones):
        if oraciones.count(o) >= 3:
            flags.append("repetitivo")
            break
    return flags


def condiciones_en(texto: str) -> set[str]:
    """Diagnósticos del léxico presentes en el texto."""
    tn = _norm(texto)
    return {c for c in CONDICIONES if _contiene_palabra(c, tn)}


def diagnosticos_no_sustentados(borrador: str, evidencia_textos: list[str]) -> list[str]:
    """
    Diagnósticos nombrados en el borrador que no aparecen en NINGUNA evidencia.
    Esos son los candidatos a 'alucinación' que el revisor debe mirar primero.
    """
    en_borrador = condiciones_en(borrador)
    en_evidencia: set[str] = set()
    for ev in evidencia_textos:
        en_evidencia |= condiciones_en(ev)
    return sorted(en_borrador - en_evidencia)


def terminos_riesgo(borrador: str) -> list[dict[str, str]]:
    tn = _norm(borrador)
    hallados: list[dict[str, str]] = []
    for categoria, terminos in TERMINOS_RIESGO.items():
        for term in terminos:
            if _contiene_palabra(term, tn):
                hallados.append({"termino": term, "categoria": categoria})
    return hallados


def recomendaciones_en(texto: str) -> set[str]:
    """Estudios/acciones del léxico presentes en el texto."""
    tn = _norm(texto)
    return {r for r in RECOMENDACIONES if _contiene_palabra(r, tn)}


def recomendaciones_no_sustentadas(borrador: str, evidencia_textos: list[str]) -> list[str]:
    """
    Estudios o tratamientos sugeridos en el borrador que no aparecen en NINGUNA
    evidencia (segundo modo de falla del paper: recomendaciones inventadas).
    """
    en_borrador = recomendaciones_en(borrador)
    en_evidencia: set[str] = set()
    for ev in evidencia_textos:
        en_evidencia |= recomendaciones_en(ev)
    return sorted(en_borrador - en_evidencia)


def cambio_de_entidad(borrador: str, evidencia_textos: list[str]) -> bool:
    """
    ¿El diagnóstico del borrador difiere del caso más parecido?

    Primer modo de falla del paper: el modelo mantiene el estilo pero desplaza la
    entidad diagnóstica central. Se marca cuando el borrador y el caso más parecido
    (primera evidencia) nombran diagnósticos y no comparten NINGUNO.
    """
    if not evidencia_textos:
        return False
    dx_borrador = condiciones_en(borrador)
    dx_top = condiciones_en(evidencia_textos[0])
    return bool(dx_borrador and dx_top and not (dx_borrador & dx_top))


def _nivel(
    flags: list[str],
    no_sustentados: list[str],
    riesgos: list[dict],
    recs_no_sustentadas: list[str],
    entidad_cambiada: bool,
) -> str:
    if "vacio" in flags or no_sustentados or entidad_cambiada or any(
        r["categoria"] in ("procedimiento_invasivo", "farmaco_sistemico") for r in riesgos
    ):
        return "alto"
    if flags or riesgos or recs_no_sustentadas:
        return "medio"
    return "bajo"


def analizar(borrador: str, evidencia_textos: list[str]) -> dict[str, Any]:
    flags = flags_heuristicos(borrador)
    no_sustentados = diagnosticos_no_sustentados(borrador, evidencia_textos)
    riesgos = terminos_riesgo(borrador)
    recs_no_sustentadas = recomendaciones_no_sustentadas(borrador, evidencia_textos)
    entidad_cambiada = cambio_de_entidad(borrador, evidencia_textos)
    return {
        "nivel": _nivel(flags, no_sustentados, riesgos, recs_no_sustentadas, entidad_cambiada),
        "flags": flags,
        "diagnosticos_no_sustentados": no_sustentados,
        "recomendaciones_no_sustentadas": recs_no_sustentadas,
        "cambio_de_entidad": entidad_cambiada,
        "terminos_riesgo": riesgos,
    }
