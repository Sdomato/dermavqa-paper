# Evaluation plan

## Objetivo

Evaluar si cada metodo produce respuestas utiles y comparables contra las
referencias del dataset, sin presentar el sistema como diagnostico clinico.

## Metricas de generacion

- ROUGE-L para solapamiento secuencial.
- Token-level F1 para precision y recall lexical.
- BERTScore multilingue para similitud semantica.
- Exact Match solo como referencia secundaria.

## Metricas de recuperacion

- Recall@k cuando el caso original pueda tratarse como target de recuperacion.
- Mean Reciprocal Rank.
- Similitud semantica entre respuesta recuperada y referencia.
- Analisis por modalidad: texto, imagen y fusion multimodal.

## Evaluacion cualitativa

Revisar manualmente una muestra pequena de casos de valid/test. Etiquetar:

- respuesta generica;
- diagnostico no respaldado;
- recomendacion insegura;
- contradiccion con la consulta;
- contradiccion con la imagen;
- caso donde la imagen aporta informacion clave;
- caso donde el texto domina y la imagen no ayuda.

Para diagnosticar diferencias entre las cuatro condiciones VLM sobre el dataset
enriquecido, usar además el protocolo contrastivo de
`survey/contrastive_error_analysis.md`. Las explicaciones generadas son
justificaciones post-hoc basadas en evidencia observable, no trazas fieles de
cadena de pensamiento.

## Comparaciones principales

1. Recuperacion solo-texto vs recuperacion solo-imagen.
2. Recuperacion solo-texto vs recuperacion multimodal.
3. VLM zero-shot vs VLM LoRA/QLoRA.
4. Recuperacion multimodal vs VLM LoRA/QLoRA.
5. Hibrido opcional vs sus componentes individuales.

## Salidas recomendadas

- `outputs/predictions/*.jsonl` con predicciones por metodo.
- `outputs/metrics/*.csv` con metricas agregadas.
- `outputs/error_analysis/*.csv` con revision cualitativa.
- tabla final lista para paper.
