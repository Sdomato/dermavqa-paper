# Team execution plan

## Objetivo general

Comparar estrategias para Visual Question Answering dermatologico en espanol
usando DermaVQA-IIYI bajo restricciones realistas de datos y computo.

La pregunta principal del trabajo es:

> Para esta tarea, conviene mas usar recuperacion multimodal sobre casos
> similares o fine-tunear un VLM liviano con LoRA/QLoRA?

El trabajo se organiza alrededor de tres variantes de dataset:

- `dataset_enriched`: respuestas sintetizadas por LLM a partir de todas las
  respuestas originales.
- `dataset_longest_answer`: respuesta original mas larga como target.
- `dataset_short_answer`: respuesta corta o etiqueta diagnostica como target.

## Estado actual

- El dataset enriquecido ya esta generado y empaquetado en
  `outputs/datasets/dermavqa_iiyi_llm_synthesized_answer_finetune.zip`.
- El baseline de retrieval textual sobre `dataset_enriched` ya esta corrido.
- Los resultados resumidos estan en
  `results/dataset_enriched/retrieval_textual/`.
- Las imagenes no se suben a GitHub porque el equipo ya las tiene localmente.

## Responsabilidades por participante

### Santino

Foco principal: dataset enriquecido y LoRA/QLoRA sobre dataset enriquecido.

Tareas:

- Mantener y documentar `dataset_enriched`.
- Validar splits, conteos, respuestas vacias, rutas de imagen y duplicados.
- Documentar como se genero el dataset enriquecido:
  - modelo usado;
  - prompt;
  - temperatura;
  - reglas de sintesis;
  - limitaciones;
  - costo aproximado.
- Mantener el baseline textual ya creado sobre `dataset_enriched`.
- Agregar visualizaciones al baseline textual:
  - `chrF`;
  - `sacreBLEU`;
  - BERTScore;
  - ganador en valid segun `chrF`.
- Fine-tunear un VLM con LoRA/QLoRA sobre `dataset_enriched`.
- Guardar resultados de fine-tuning en
  `results/dataset_enriched/vlm_lora/`.

Entregables:

- Dataset enriquecido documentado.
- Notebook/resultados de retrieval textual enriquecido con plots.
- Notebook de LoRA/QLoRA sobre dataset enriquecido.
- Predicciones y metricas en valid/test.

### Damian

Foco principal: datasets de respuesta larga/corta, baselines y retrieval visual.

Tareas:

- Crear `dataset_longest_answer`, usando la respuesta original mas larga.
- Crear `dataset_short_answer`, usando una respuesta corta o diagnostico breve.
- Implementar todos los baselines para `dataset_longest_answer`:
  - retrieval textual TF-IDF;
  - retrieval textual Multilingual E5;
  - retrieval textual Sentence-BERT multilingue;
  - retrieval visual con CLIP/OpenCLIP;
  - retrieval multimodal texto + imagen.
- Implementar retrieval visual para `dataset_short_answer`.
- Agregar graficos al retrieval textual que Santino ya hizo sobre
  `dataset_enriched`.

Resultados esperados:

- `results/dataset_longest_answer/retrieval_textual/`
- `results/dataset_longest_answer/retrieval_visual/`
- `results/dataset_longest_answer/retrieval_multimodal/`
- `results/dataset_short_answer/retrieval_visual/`
- plots agregados en `results/dataset_enriched/retrieval_textual/`

### Matias

Foco principal: VLM sobre dataset de respuesta larga y escritura del paper.

Tareas:

- Trabajar con `dataset_longest_answer`.
- Implementar VLM zero-shot:
  - input: imagen + pregunta;
  - output: respuesta larga original.
- Fine-tunear con LoRA/QLoRA sobre `dataset_longest_answer` si el computo lo
  permite.
- Comparar:
  - VLM zero-shot sobre respuesta larga;
  - VLM LoRA/QLoRA sobre respuesta larga;
  - retrieval multimodal sobre respuesta larga.
- Liderar la escritura del paper:
  - motivacion;
  - datasets;
  - metodos;
  - resultados;
  - limitaciones;
  - conclusion retrieval vs fine-tuning.

Resultados esperados:

- `results/dataset_longest_answer/vlm_zero_shot/`
- `results/dataset_longest_answer/vlm_lora/`
- tablas y texto base para el paper.

## Metricas para comparar modelos fine-tuned

Santino y Matias deben calcular las mismas metricas para que la comparacion sea
justa.

### Metricas lexicas

- `sacreBLEU`: overlap estricto de n-gramas.
- `chrF`: overlap por caracteres; metrica automatica principal para espanol.
- `ROUGE-L`: overlap de subsecuencia mas larga.
- `token-level F1`: precision, recall y F1 sobre tokens normalizados.

### Metricas semanticas

- BERTScore F1 multilingue.
- Cosine similarity entre embeddings de prediccion y referencia usando E5 o
  Sentence-BERT.

### Metricas clinicas manuales

Calcular sobre una muestra manual de 10 a 20 casos por metodo:

- `diagnosis_supported_rate`: diagnostico respaldado por referencia/pregunta.
- `unsafe_recommendation_rate`: recomendaciones no respaldadas o riesgosas.
- `hallucination_rate`: informacion clinica agregada sin soporte.
- `too_generic_rate`: respuestas demasiado vagas.
- `empty_or_invalid_rate`: respuestas vacias, truncadas o que repiten prompt.

### Metricas operativas

- Tiempo promedio de inferencia por ejemplo.
- Tiempo total de fine-tuning.
- VRAM usada durante entrenamiento/inferencia.
- Tamano del adapter LoRA.
- Costo aproximado si se usa cloud.

## Protocolo comun de evaluacion

- Entrenar solo con `train`.
- Usar `valid` para seleccionar hiperparametros/checkpoint.
- Usar `test` solo para el reporte final.
- Generar siempre:
  - `predictions_valid.csv`;
  - `predictions_test.csv`;
  - `metrics_valid.csv`;
  - `metrics_test.csv`;
  - `manual_review_20.csv`.
- Guardar predicciones con al menos:
  - `split`;
  - `encounter_id`;
  - `image_id`;
  - `question_es`;
  - `reference_answer_es`;
  - `predicted_answer_es`;
  - `model_name`;
  - `dataset_variant`;
  - `method`.

## Comparacion final Santino vs Matias

Comparacion principal:

- Santino: LoRA/QLoRA sobre `dataset_enriched`.
- Matias: LoRA/QLoRA sobre `dataset_longest_answer`.

Cada modelo se evalua primero contra su propio target:

- modelo de Santino contra `answer_es` enriquecida;
- modelo de Matias contra `longest_answer`.

Comparacion cruzada opcional:

- modelo enriquecido contra referencias de respuesta larga;
- modelo de respuesta larga contra referencias enriquecidas.

La decision final no debe basarse solo en una metrica automatica. Se elige el
mejor enfoque combinando:

- `chrF` y BERTScore;
- menor hallucination/unsafe rate;
- mejor revision manual;
- costo y tiempo razonables;
- claridad de la respuesta para un asistente dermatologico.

## Estructura esperada de resultados

```text
results/
  dataset_enriched/
    retrieval_textual/
    vlm_lora/
  dataset_longest_answer/
    retrieval_textual/
    retrieval_visual/
    retrieval_multimodal/
    vlm_zero_shot/
    vlm_lora/
  dataset_short_answer/
    retrieval_visual/
```

## Reglas de colaboracion

- No subir imagenes a GitHub.
- No subir claves, `.env`, checkpoints pesados ni caches.
- Subir notebooks, scripts, metricas resumidas y documentos del survey.
- Si un resultado pesa mucho, subir solo metricas agregadas y documentar donde
  conseguir el artefacto completo.
- Mantener nombres de carpetas consistentes por dataset y metodo.

