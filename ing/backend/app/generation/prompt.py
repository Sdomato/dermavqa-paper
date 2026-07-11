"""
Armado del prompt RAG (Retrieval-Augmented Generation).

Función pura y testeable: toma la consulta del paciente y los casos similares
recuperados, y arma el texto que se le pasa al VLM. La instrucción es
deliberadamente conservadora — anclar la respuesta en los casos de referencia y
NO inventar — porque el paper mostró que el modelo, suelto, alucina diagnósticos
y tratamientos no sustentados.
"""

from typing import Any

_INSTRUCCION = (
    "Sos un asistente que redacta un BORRADOR de respuesta dermatológica para que "
    "un médico lo revise y apruebe. Basate ÚNICAMENTE en los casos similares de "
    "referencia (y en la imagen, si está). No inventes diagnósticos, estudios ni "
    "tratamientos que no estén respaldados por la evidencia. Si los casos son "
    "insuficientes o se contradicen, decilo explícitamente en lugar de inventar."
)

_CIERRE = (
    "Redactá un borrador conciso en español, fiel a los casos de referencia. "
    "Recordá: es un borrador para revisión médica, no una indicación final."
)


def build_rag_prompt(query: str, evidence: list[dict[str, Any]]) -> str:
    """
    `query`: consulta del paciente (título + contenido).
    `evidence`: lista de casos recuperados; cada uno con al menos la clave 'answer'
                (y opcionalmente 'similitud').
    """
    lines = [_INSTRUCCION, "", f"CONSULTA DEL PACIENTE:\n{query.strip()}", ""]
    if evidence:
        lines.append("CASOS SIMILARES DE REFERENCIA:")
        for i, ev in enumerate(evidence, 1):
            ans = str(ev.get("answer", "")).strip()
            sim = ev.get("similitud")
            cab = f"[{i}]" + (f" (similitud {sim:.2f})" if isinstance(sim, (int, float)) else "")
            lines.append(f"{cab} {ans}")
    else:
        lines.append("(No se encontraron casos similares de referencia.)")
    lines += ["", _CIERRE]
    return "\n".join(lines)
