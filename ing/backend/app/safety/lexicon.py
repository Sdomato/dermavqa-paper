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
    "granuloma", "granuloma anular", "onicomicosis", "necrobiosis lipoidea",
    "sindrome de sweet", "penfigoide", "lupus", "escarlatina",
    "erupcion por medicamentos", "erupcion medicamentosa", "farmacodermia",
    "erupcion fija", "exantema", "picadura de insecto",
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

# Estudios / acciones que el borrador puede recomendar. Se usan para el chequeo de
# recomendaciones no sustentadas: si el borrador sugiere un estudio o tratamiento
# que NO aparece en la evidencia, es el segundo modo de falla del paper (el modelo
# propone biopsias, análisis o tratamientos ausentes en la referencia).
RECOMENDACIONES = [
    "biopsia", "cultivo", "serologia", "dermatoscopia",
    "prueba de hongos", "prueba de patologia", "prueba de alergenos",
    "prueba de sangre", "pruebas de sangre", "analisis de sangre", "examen de sangre",
    "prueba de funcion hepatica",
    "antihistaminico", "antihistaminicos", "antifungico", "antifungicos",
    "antimicotico", "antibiotico", "antibioticos",
    "crioterapia", "laser", "fototerapia",
    "derivar", "derivacion", "interconsulta",
]
