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

# Banderas rojas: señales de malignidad / urgencia que se buscan en la CONSULTA del
# paciente (no en el borrador). Si aparecen, el caso se fuerza a nivel alto sin
# importar qué diga el borrador: la seguridad parte del paciente, no del modelo.
#
# Cada regla es (nombre_del_signo, [grupos]); un grupo dispara si TODOS sus términos
# están presentes (permite exigir co-ocurrencia, ej. "lunar" + "sangra").
BANDERAS_ROJAS = [
    ("sospecha_melanoma", [["melanoma"]]),
    ("mencion_cancer", [["cancer"], ["carcinoma"], ["metastasis"], ["maligno"], ["maligna"]]),
    ("lesion_pigmentada_cambiante", [
        ["lunar", "cambio"], ["lunar", "cambia"], ["lunar", "crece"], ["lunar", "crecio"],
        ["lunar", "sangra"], ["lunar", "asimetric"], ["lunar", "color"],
        ["mancha", "sangra"], ["mancha", "crece"], ["mancha", "negro"], ["mancha", "negra"],
        ["nevo", "cambio"], ["nevo", "crece"], ["nevo", "sangra"],
        ["lesion pigmentada", "cambio"], ["lesion pigmentada", "crece"],
        ["pigmentada", "asimetric"],
    ]),
    ("lesion_no_cicatriza", [
        ["no cicatriza"], ["no cierra"], ["no cura"], ["no termina de cerrar"],
        ["ulcera", "meses"], ["herida", "no cierra"],
    ]),
    ("lesion_acral", [["acral"], ["planta del pie"], ["subungueal"], ["region palmoplantar"]]),
    ("crecimiento_rapido", [
        ["crecio", "semanas"], ["crecio", "dias"], ["crece rapido"], ["rapido crecimiento"],
        ["crecimiento rapido"], ["creciendo rapido"],
    ]),
    ("sangrado", [["sangra"], ["sangrado"], ["sangrante"]]),
    ("urgencia_sistemica", [
        ["fiebre", "ampollas"], ["fiebre", "despega"], ["piel se despega"],
        ["dificultad", "respirar"], ["se despega la piel"], ["necrosis"],
    ]),
]

# Frases del borrador que tranquilizan/descartan. Si aparecen y la consulta tiene
# banderas rojas, es una falsa tranquilización: escalar y exigir revisión.
FRASES_TRANQUILIZADORAS = [
    "se puede descartar", "descartar la posibilidad", "se descarta",
    "no es preocupante", "no reviste gravedad", "no es grave", "sin gravedad",
    "es benigno", "es benigna", "no hay de que preocuparse", "quedese tranquilo",
    "no requiere tratamiento", "no amerita",
]
