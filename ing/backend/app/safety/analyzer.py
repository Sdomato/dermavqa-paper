"""
Análisis de seguridad de un borrador (Fase 3).

Todo es lógica pura y determinística (testeable sin modelos):

1. Heurísticos      → vacío / muy corto / repetitivo.
2. Grounding        → diagnósticos nombrados en el borrador que NO aparecen en la
                      evidencia recuperada (el modo de falla del paper: el modelo
                      desplaza la entidad diagnóstica).
3. Términos de riesgo → recomendaciones sensibles (biopsia, antibióticos, etc.).
4. Nivel global     → low / medium / high, para priorizar la revisión médica.

El objetivo es SEÑALAR para el revisor humano, no decidir clínicamente.
"""

import re
import unicodedata
from typing import Any

from .lexicon import CONDICIONES, TERMINOS_RIESGO

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


def _nivel(flags: list[str], no_sustentados: list[str], riesgos: list[dict]) -> str:
    if "vacio" in flags or no_sustentados or any(
        r["categoria"] in ("procedimiento_invasivo", "farmaco_sistemico") for r in riesgos
    ):
        return "alto"
    if flags or riesgos:
        return "medio"
    return "bajo"


def analizar(borrador: str, evidencia_textos: list[str]) -> dict[str, Any]:
    flags = flags_heuristicos(borrador)
    no_sustentados = diagnosticos_no_sustentados(borrador, evidencia_textos)
    riesgos = terminos_riesgo(borrador)
    return {
        "nivel": _nivel(flags, no_sustentados, riesgos),
        "flags": flags,
        "diagnosticos_no_sustentados": no_sustentados,
        "terminos_riesgo": riesgos,
    }
