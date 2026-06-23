"""
Léxicos del dominio dermatológico para la capa de seguridad.

- CONDICIONES: diagnósticos/entidades clínicas frecuentes en el dataset. Se usan
  para el chequeo de grounding (¿el borrador nombra un diagnóstico que no está en
  la evidencia?).
- TERMINOS_RIESGO: recomendaciones que ameritan escrutinio extra (procedimientos
  invasivos, fármacos de cuidado). No significa "malo", significa "que el médico
  lo mire con atención".

Listas acotadas y editables a propósito: el objetivo es señalar, no diagnosticar.
"""

# Diagnósticos / entidades clínicas (forma normalizada sin acentos, en minúscula).
CONDICIONES = [
    "psoriasis", "eccema", "eczema", "dermatitis", "dermatitis de contacto",
    "dermatitis seborreica", "dermatitis atopica", "urticaria", "liquen plano",
    "tina", "tinea", "foliculitis", "sifilis", "dishidrosis", "linfangioma",
    "nevus", "nevo", "angioma", "verruga", "acne", "melanoma", "carcinoma",
    "vitiligo", "rosacea", "impetigo", "herpes", "escabiosis", "sarna",
    "alopecia", "paroniquia", "celulitis", "molusco", "queratosis",
    "pitiriasis", "intertrigo", "candidiasis", "eritema multiforme",
    "sindrome de sweet", "penfigoide", "lupus", "escarlatina",
]

# Recomendaciones de riesgo, por categoría.
TERMINOS_RIESGO = {
    "procedimiento_invasivo": ["biopsia", "cirugia", "extirpacion", "infiltracion", "extraccion"],
    "farmaco_sistemico": [
        "antibiotico", "antibioticos", "corticoide", "corticoides", "corticosteroide",
        "isotretinoina", "metotrexato", "inmunosupresor", "prednisona",
    ],
    "tratamiento_hormonal": ["hormona", "hormonal", "estrogeno", "testosterona"],
}
