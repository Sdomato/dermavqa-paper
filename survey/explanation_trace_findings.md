# Análisis de explicaciones post-hoc y métricas VLM

## Objetivo

Este documento conecta las métricas de los modelos con patrones observables en
sus respuestas y en los campos estructurados de explicación:

- `explanation`;
- `candidate_support`;
- `visual_evidence_es`;
- `question_evidence_es`;
- `uncertainty_es`;
- `rag_context_use_es`.

La meta es explicar por qué una condición obtuvo mejores o peores métricas sin
presentar estas explicaciones como una cadena de pensamiento real del modelo.

## Qué representan las explicaciones

Se generaron dos tipos de artefactos:

1. **Self explanations:** se volvió a inferir con un prompt que pedía respuesta
   y explicación. Ese cambio de prompt alteró las respuestas y los adapters
   LoRA casi nunca respetaron el JSON. No son comparables con las predicciones
   usadas para calcular las métricas.
2. **External rationales:** un analista VLM común recibió imagen, pregunta,
   contexto RAG cuando correspondía y la predicción original fija. Su tarea fue
   justificar o cuestionar esa predicción sin recibir el ground truth.

Para este análisis usamos principalmente los external rationales porque
mantienen exactamente la predicción evaluada. Son justificaciones post-hoc
comparables, no evidencia del mecanismo interno de cada modelo.

Cobertura:

| Dataset | Casos contrastivos | Métodos | Racionales | Parseados |
| --- | ---: | ---: | ---: | ---: |
| Enriched | 25 imágenes | 5 | 125 | 123 |
| Longest answer | 23 imágenes | 4 | 92 | 90 |

La muestra fue seleccionada por ganancias, pérdidas, desacuerdo entre métricas,
longitudes extremas e inconsistencia multiimagen. No es una muestra aleatoria y
sus porcentajes descriptivos no estiman la frecuencia de errores en todo test.

## Resultados globales

### Dataset enriched

| Método | chrF | ROUGE-L | Token F1 | BERTScore F1 |
| --- | ---: | ---: | ---: | ---: |
| Zero-shot | 0.298 | 0.123 | 0.189 | 0.680 |
| Zero-shot + RAG | 0.310 | 0.136 | 0.199 | 0.682 |
| LoRA | **0.365** | **0.254** | **0.317** | **0.738** |
| LoRA + RAG en inferencia | 0.302 | 0.228 | 0.274 | 0.724 |
| LoRA RAG-aware | 0.298 | 0.248 | 0.288 | 0.737 |

Efectos pareados sobre las 314 filas de test:

- LoRA mejora el score compuesto frente a zero-shot en 96.5% de las filas,
  con delta medio `+0.096`.
- RAG sobre zero-shot tiene un efecto pequeño: `+0.009`, con 55.1% de filas
  positivas.
- RAG agregado al LoRA solo en inferencia produce `-0.036` y mejora 33.8%.
- La ventaja de LoRA es mayor con referencias cortas (`+0.153`) que con
  referencias de más de 60 palabras (`+0.061`).

### Dataset longest answer

| Método | chrF | ROUGE-L | Token F1 | BERTScore F1 |
| --- | ---: | ---: | ---: | ---: |
| Zero-shot | **0.211** | **0.083** | **0.110** | **0.646** |
| Zero-shot + RAG | 0.202 | 0.083 | 0.107 | 0.642 |
| LoRA | 0.168 | 0.081 | 0.092 | 0.619 |
| LoRA + RAG en inferencia | 0.121 | 0.041 | 0.033 | 0.560 |

Efectos pareados:

- LoRA frente a zero-shot produce delta medio `-0.022` y mejora solo 39.2%.
- En referencias largas el delta de LoRA cae a `-0.103`; solo 6.25% mejora.
- RAG sobre LoRA produce `-0.051` y mejora apenas 25.2%.
- LoRA es más competitivo en referencias cortas y encuentros con una imagen.

## Hallazgo central: el target enriquecido mejora la alineación

La explicación más consistente de las métricas no es que el LoRA enriched haya
aprendido una visión dermatológica radicalmente superior. Los campos
`visual_evidence_es` de todos los métodos siguen siendo frecuentemente
genéricos: “lesión”, “ampolla”, “erupción” o una repetición de la consulta.

La diferencia aparece principalmente en la forma de convertir esa evidencia en
una respuesta:

- zero-shot enriched tiende a respuestas extensas, negativas a diagnosticar,
  recomendaciones generales y diferenciales abiertos;
- LoRA enriched adopta el patrón estable del target: “El cuadro es compatible
  con...”, seguido de uno o dos diferenciales y una prueba o recomendación;
- la incertidumbre queda alrededor de una hipótesis clínica concreta en vez de
  reemplazarla por una abstención genérica.

En la muestra contrastiva, la referencia enriched tiene 32.6 palabras en
promedio. Zero-shot genera unas 100 palabras, mientras LoRA genera 51.6. LoRA se
acerca más a la longitud y estructura esperadas, lo que explica la mejora
simultánea de chrF, ROUGE-L y token-F1. BERTScore también sube porque las
respuestas dejan de ser solo más parecidas superficialmente: contienen con
mayor frecuencia la entidad diagnóstica y el manejo presentes en la referencia.

Esto debe describirse como **mejor alineación con el target enriquecido**, no
como demostración automática de mayor corrección clínica.

### Caso representativo: `ENC00916`

- Ground truth: tinea versicolor, con eritrasma como diferencial.
- Zero-shot: deriva a varicela o se limita a decir que no puede diagnosticar.
- LoRA: responde “infección fúngica, probablemente tiña versicolor” y propone
  confirmación micológica.
- LoRA + RAG: cambia a psoriasis y rosácea, aunque la evidencia actual no
  justifica ese salto.
- LoRA RAG-aware: vuelve a una hipótesis fúngica, aunque usa “tiña corporal”.

Este caso resume las métricas: LoRA aprende el espacio de respuestas del target;
RAG simple puede desplazarlo hacia la etiqueta dominante de otro caso; el
entrenamiento RAG-aware reduce parte de esa interferencia.

### Excepción: `ENC00965`

La referencia menciona nevo epidérmico lineal o habón. LoRA responde eccema,
psoriasis y dermatitis seborreica; RAG introduce incluso una erupción por
medicamentos. Aunque la redacción mantiene el estilo del dataset, la entidad
diagnóstica es incorrecta. Esto demuestra que el patrón textual que mejora las
métricas no garantiza exactitud clínica.

## Por qué zero-shot gana en longest answer

El target longest es mucho más heterogéneo. En la muestra contrastiva tiene 87.4
palabras de promedio, pero combina etiquetas breves con narrativas extensas,
tratamientos detallados y estilos distintos.

Zero-shot genera aproximadamente 101 palabras y conserva más antecedentes,
síntomas, advertencias y explicación general. Esa cobertura se aproxima mejor a
las referencias largas y favorece chrF y BERTScore, aunque el diagnóstico pueda
seguir siendo impreciso.

LoRA longest genera unas 70 palabras en promedio y muestra dos problemas:

1. **Pérdida de cobertura:** reduce respuestas largas a una etiqueta o frase
   breve, por lo que omite términos presentes en la referencia.
2. **Degeneración de formato:** en varios casos produce listas de preguntas o
   repite el mismo diagnóstico con pequeñas variaciones.

Ambos fenómenos reducen ROUGE-L y token-F1. Cuando la repetición también cambia
la entidad diagnóstica, cae BERTScore.

### Casos representativos

- `ENC00985`: la referencia desarrolla urticaria colinérgica y su tratamiento;
  LoRA responde solo “Urticaria Papular”. Pierde cobertura aunque la familia
  diagnóstica sea cercana. LoRA + RAG repite una lista de variantes de urticaria
  papular, aumentando longitud sin recuperar información útil.
- `ENC00988`: la referencia breve sugiere verruga. LoRA repite variantes de
  queratosis seborreica/folicular; RAG cambia a una lista de tipos de verruga,
  pero sin una conclusión estable.
- `ENC00940`: distintas imágenes del mismo encuentro producen negativa,
  hemangioma, nevus, glándula sebácea o listas repetidas. La variabilidad muestra
  que entrenar cada imagen contra el mismo target no garantiza consistencia
  entre vistas.

## Qué ocurre al agregar RAG

### RAG sobre el modelo base

RAG ayuda cuando reemplaza una abstención por una respuesta concreta o aporta
una entidad presente en la referencia. Por eso mejora ligeramente zero-shot en
enriched. El efecto es pequeño porque también puede introducir un vecino
semánticamente parecido pero clínicamente distinto.

Los campos `rag_context_use_es` suelen ser genéricos y a veces conservan la
plantilla “como influyeron los casos recuperados, o no_aplica”. Por ello no son
prueba de atención real al contexto. Los ejemplos sí muestran cambios de
contenido compatibles con arrastre desde los recuperados.

### RAG sobre LoRA solo en inferencia

El LoRA enriched ya posee un prior fuerte y alineado con el target. Agregarle
casos de otros pacientes cambia diagnósticos, tratamientos o longitud sin que el
modelo haya sido entrenado para decidir qué ignorar.

Ejemplos:

- `ENC00916`: tiña versicolor pasa a psoriasis/rosácea.
- `ENC01001`: adopta neurodermatitis y tratamientos de un vecino con prurito,
  aunque la consulta actual declara ausencia de picazón.
- `ENC00973`: urticaria papular cambia a pitiriasis rosada.

En longest se suma la inestabilidad del adapter: el contexto amplía listas,
introduce entidades nuevas y favorece repeticiones. Por eso LoRA + RAG obtiene
las peores métricas.

### LoRA RAG-aware

Entrenar enriched con el mismo prompt RAG usado en inferencia mejora respecto de
LoRA + RAG: BERTScore vuelve de 0.724 a 0.737 y ROUGE-L de 0.228 a 0.248. Los
casos `ENC00916`, `ENC00940` y `ENC00973` muestran que puede volver a una familia
diagnóstica más compatible después de la desviación causada por RAG simple.

Sin embargo, sus respuestas son más cortas: 27.9 palabras frente a 32.6 de la
referencia y 51.6 de LoRA. Esa concisión conserva significado general, pero
omite detalles y reduce chrF y token-F1. Además, todavía puede incorporar
tratamientos o diagnósticos del contexto, como ocurre en `ENC01001`.

## Qué se puede afirmar en el paper

1. **La mejora de LoRA enriched está asociada a una señal supervisada más
   homogénea.** Sus respuestas son más directas, menos evasivas y más cercanas
   a la longitud, vocabulario y estructura de las referencias.
2. **El fine-tuning no mejora universalmente el razonamiento clínico.** Sobre
   longest, LoRA pierde cobertura y presenta repetición; sobre enriched también
   existen errores diagnósticos con redacción convincente.
3. **RAG textual aporta utilidad limitada.** Ayuda al modelo base cuando evita
   una respuesta genérica, pero puede transferir diagnósticos o tratamientos de
   otro paciente.
4. **La recuperación de RAG-aware es consistente con que el desajuste entre
   entrenamiento e inferencia contribuya al daño de RAG sobre LoRA.** Aun así,
   no supera al LoRA enriched sin RAG.
5. **La calidad del target domina la comparación.** Un objetivo consistente
   permite que un adapter pequeño aprenda mejor el formato y el espacio de
   respuestas; una respuesta “más larga” no es necesariamente una supervisión
   más informativa.

## Qué no se puede afirmar

- Que `explanation` sea el razonamiento interno o una chain-of-thought real.
- Que una explicación plausible valide clínicamente la predicción.
- Que `candidate_support=supported` implique coincidencia con el ground truth:
  el analista externo no recibió la referencia y a veces justifica respuestas
  incorrectas.
- Que los porcentajes de la muestra contrastiva representen todo el test.
- Que las diferencias pequeñas sean estadísticamente significativas: no se
  ejecutaron múltiples semillas ni intervalos agrupados por `encounter_id`.
- Que una métrica alta equivalga a seguridad médica.

## Texto breve sugerido para la discusión

> El análisis contrastivo sugiere que la ventaja del LoRA enriquecido proviene
> principalmente de una mejor alineación con una señal supervisada homogénea.
> Sus respuestas adoptan una estructura clínica estable, reducen abstenciones
> genéricas y preservan una hipótesis diagnóstica concreta, lo que mejora tanto
> métricas léxicas como BERTScore. En contraste, el target de respuesta más
> larga combina etiquetas breves y narrativas extensas; el adapter pierde
> cobertura y, en algunos casos, genera listas repetitivas. El contexto RAG
> textual ayuda ocasionalmente al modelo base, pero con frecuencia introduce
> diagnósticos o tratamientos de casos vecinos. Entrenar con el mismo formato
> RAG reduce ese desajuste, aunque no supera al LoRA enriquecido sin
> recuperación. Estas observaciones provienen de racionales post-hoc y deben
> interpretarse como hipótesis respaldadas por patrones de salida, no como
> trazas causales del razonamiento interno.

## Fuentes de evidencia

- `outputs/metrics/dataset_enriched/metrics_mixed.csv`;
- `outputs/metrics/dataset_longest_answer/metrics_mixed.csv`;
- `outputs/error_analysis/dataset_enriched/external_rationales_all.csv`;
- `outputs/error_analysis/dataset_longest_answer/external_rationales_all.csv`;
- `outputs/error_analysis/*/contrastive_cases_test.csv`;
- `outputs/error_analysis/*/metric_strata_summary.csv`;
- `outputs/error_analysis/*/pairwise_effect_summary.csv`;
- `outputs/error_analysis/*/external_rationale_summary.md`.
