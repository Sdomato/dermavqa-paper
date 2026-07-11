# Survey y roadmap del proyecto

## Resumen

Este proyecto estudia Visual Question Answering dermatologico en espanol usando
DermaVQA-IIYI. La pregunta central es metodologica:

> Bajo restricciones realistas de datos y computo, conviene adaptar un VLM con
> LoRA/QLoRA o usar recuperacion multimodal sobre casos dermatologicos similares?

El objetivo no es construir una herramienta clinica desplegable. El objetivo es
comparar, de forma empirica y reproducible, estrategias de adaptacion para un
dominio medico multimodal y subrepresentado en espanol.

## Documentos de esta carpeta

- `dataset_notes.md`: inventario del dataset, conteos y auditorias pendientes.
- `experiment_grid.md`: condiciones experimentales, parametros y entregables.
- `models_to_compare.md`: candidatos para retrieval textual, visual y VLM.
- `evaluation_plan.md`: metricas automaticas, retrieval y revision cualitativa.
- `contrastive_error_analysis.md`: muestreo contrastivo y explicaciones
  estructuradas para diagnosticar diferencias entre VLMs.
- `explanation_trace_findings.md`: análisis de racionales post-hoc, patrones por
  modelo y conclusiones para justificar las métricas en el paper.
- `team_execution_plan.md`: division de tareas, protocolo comun y metricas
  para comparar modelos fine-tuned.
- `enriched_vlm_experiments.md`: corrida LoRA/QLoRA de Santino sobre
  `dataset_enriched`, metricas, artefactos, observaciones y pendientes.
- `final_comparison_snapshot.md`: foto comparativa actual entre retrieval,
  VLM zero-shot, LoRA sobre respuesta larga y LoRA sobre enriquecido.
- `paper_results_interpretation.md`: lectura de resultados, claim principal,
  limitaciones y texto sugerido para el paper.
- `../paper/draft.md`: borrador narrativo completo del paper.
- `../notebooks/04_paper_results.ipynb`: regeneracion de tablas y figuras
  paper-ready desde los artefactos versionados.
- `../src/evaluate_retrieval_heldout.py`: baseline TF-IDF held-out train-only
  para evitar leakage en las comparaciones principales longest/short.

- `STRUCTURE.md`: convencion canonica de carpetas y artefactos del repo
  (raiz de resultados, que se versiona, resolucion de imagenes).
- `matias_execution_plan.md`: plan y progreso de la pata VLM (zero-shot + LoRA
  sobre `dataset_longest_answer`): `src/vlm_infer.py`, `src/train_longest.py`,
  `src/evaluate_predictions.py`.
- `google_cloud_vlm_lora_runbook.md`: protocolo reproducible en Google Cloud
  para entrenar Qwen2.5-VL-3B con LoRA/QLoRA sobre `dataset_enriched` y
  `dataset_longest_answer_by_image`, con VM L4, smoke tests, background jobs,
  monitoreo, bajada de resultados y reglas de versionado.
- `related_work_matrix.csv`: matriz inicial para organizar papers relacionados.
- `risks.md`: riesgos metodologicos, medicos y operativos.

## Contexto del repo

- `data/iiyi/` contiene el subconjunto IIYI con casos, imagenes y metadatos.
- `notebooks/01_explore.ipynb` ya explora el dataset y construye pares
  imagen-pregunta-respuesta para entrenamiento.
- `requirements.txt` incluye dependencias para EDA, fine-tuning de VLMs y
  evaluacion (`transformers`, `peft`, `accelerate`, `bert-score`, `sacrebleu`).
- `src/` contiene el pipeline modular: construccion de datasets, baselines de
  retrieval (textual/visual/multimodal), VLM (zero-shot + LoRA) y evaluacion.
- `outputs/paper/` contiene tablas y figuras SVG listas para usar en el paper.
  Ver el README raiz y `STRUCTURE.md`.
- `config.yaml` define paths relativos para datos y salidas.

Estado observado:

- 998 casos en `train.json`, `valid_ht.json` y `test_ht.json`.
- 2.945 archivos de imagen encontrados bajo `data/iiyi/images_final`.
- 2.944 imagenes estan referenciadas por los casos; hay 1 imagen extra no usada.
- `split2encounterids.json` reporta 851 casos de train, mientras `train.json`
  contiene 842. El puente es `instanceid2encounterid.json`; hay 9 IDs de train
  del split/mapa que no aparecen en los JSON/CSV finales.

## Dataset y tarea

Cada caso combina:

- una o mas imagenes dermatologicas;
- una consulta del paciente en chino, ingles y espanol;
- multiples respuestas medicas asociadas;
- metadatos de autor, ranking y validacion cuando estan disponibles.

La tarea principal sera generar o recuperar una respuesta breve en espanol dada
una consulta textual y una imagen dermatologica. Para construir targets iniciales,
el notebook propone usar la respuesta mas extensa disponible, porque las
respuestas cortas tipo etiqueta diagnostica no siempre son informativas.

## Variantes de dataset

Como cada imagen/caso puede tener varias respuestas medicas, el proyecto no debe
depender de una sola definicion de target. Vamos a preparar varias vistas del
mismo corpus:

- `longest_answer`: una fila por imagen usando la respuesta mas extensa.
- `majority_answer`: una fila por imagen usando la respuesta mayoritaria cuando
  exista acuerdo suficiente.
- `all_answers_raw`: una fila por imagen conservando todas las respuestas como
  lista, sin resumirlas.
- `llm_synthesized_answer`: una fila por imagen donde todas las respuestas se
  consolidan en un parrafo largo y coherente usando un LLM.

Para `llm_synthesized_answer` usaremos los creditos disponibles de Microsoft
Azure, idealmente mediante un endpoint de Azure OpenAI o un LLM desplegado en
Azure. La sintesis debe ser extractiva y conservadora: combinar informacion
presente en las respuestas originales, no agregar diagnosticos ni
recomendaciones nuevas. Cada salida debe guardar el modelo, prompt, temperatura,
fecha, respuestas originales y version del script para mantener trazabilidad.

## Lineas de trabajo con subagentes

### 1. Subagente de datos

Responsable de convertir la exploracion del notebook en una preparacion
reproducible.

Tareas:

- auditar splits, conteos de imagenes y casos sin imagen;
- generar un CSV/JSONL final con rutas, preguntas, targets y metadatos;
- documentar reglas de seleccion de target: respuesta mas larga, respuesta
  mayoritaria o variantes filtradas por acuerdo;
- construir variantes de dataset, incluyendo la version sintetizada por LLM en
  Azure a partir de todas las respuestas disponibles;
- registrar estadisticas basicas: longitud de preguntas, longitud de respuestas,
  numero de imagenes por caso y acuerdo entre respuestas.

Entregables:

- script de preparacion en `src/`;
- dataset procesado en `outputs/`;
- reporte corto de estadisticas.

### 2. Subagente de recuperacion textual

Responsable de los baselines que usan solo la consulta del paciente.

Tareas:

- implementar TF-IDF como baseline simple;
- implementar embeddings multilingues, por ejemplo multilingual E5 o
  Sentence-BERT;
- recuperar top-k casos similares desde train para cada ejemplo de valid/test;
- devolver la respuesta asociada al caso recuperado.

Entregables:

- ranking top-k por consulta;
- respuestas recuperadas;
- metricas de recuperacion y generacion.

### 3. Subagente de recuperacion visual

Responsable de los baselines que usan solo imagen.

Tareas:

- codificar imagenes con un encoder visual o vision-lenguaje preentrenado;
- evaluar opciones como CLIP, OpenCLIP o BiomedCLIP segun disponibilidad;
- recuperar imagenes similares del conjunto de train;
- mapear imagen recuperada a caso y respuesta.

Entregables:

- indice visual;
- ranking top-k por imagen;
- analisis de casos donde la imagen aporta informacion no presente en el texto.

### 4. Subagente multimodal

Responsable de fusionar texto e imagen durante recuperacion.

Tareas:

- normalizar scores textual y visual;
- combinar similitudes con `s = alpha * s_text + (1 - alpha) * s_image`;
- barrer valores de `alpha`;
- comparar recuperacion solo-texto, solo-imagen y multimodal.

Entregables:

- tabla por `alpha`;
- mejor configuracion validada;
- comparacion final en test.

### 5. Subagente VLM

Responsable de zero-shot y fine-tuning liviano.

Tareas:

- correr un VLM open-source en zero-shot sobre la tarea en espanol;
- preparar prompts consistentes para imagen + consulta;
- fine-tunear con LoRA/QLoRA si el computo disponible lo permite;
- considerar modelos pequenos o medianos como Qwen2.5-VL-3B, PaliGemma u otro
  VLM compatible con la GPU disponible.

Entregables:

- predicciones zero-shot;
- checkpoint/adapters LoRA;
- predicciones fine-tuned;
- notas de costo y viabilidad.

Estado actual: Qwen2.5-VL-3B ya fue corrido en zero-shot/LoRA sobre
`dataset_longest_answer` y en LoRA sobre `dataset_enriched`.

### 6. Subagente de evaluacion

Responsable de comparar condiciones con metricas automaticas y revision manual.

Metricas:

- ROUGE-L;
- token-level F1;
- BERTScore multilingue;
- Recall@k y MRR para recuperacion;
- inspeccion cualitativa de seguridad y errores.

Errores a etiquetar:

- respuesta demasiado generica;
- diagnostico no respaldado;
- recomendacion insegura;
- contradiccion con imagen o texto;
- caso donde la imagen cambia la respuesta esperada.

### 7. Subagente de paper

Responsable de mantener la narrativa ACL alineada con los experimentos.

Tareas:

- actualizar motivacion, dataset, metodo y resultados;
- mantener una tabla clara de comparaciones principales;
- escribir limitaciones y alcance no clinico;
- conectar resultados con la pregunta LoRA vs recuperacion multimodal.

## Comparaciones principales

1. Recuperacion solo-texto vs recuperacion solo-imagen.
2. Recuperacion solo-texto vs recuperacion multimodal.
3. VLM zero-shot vs VLM fine-tuneado con LoRA/QLoRA.
4. Recuperacion multimodal vs fine-tuning liviano.
5. Modelo hibrido opcional vs enfoques individuales.

## Roadmap sugerido

1. Pasar el notebook exploratorio a scripts reproducibles. **Hecho.**
2. Auditar splits, conteos y targets del dataset. **Hecho.**
3. Implementar baseline textual TF-IDF. **Hecho.**
4. Implementar baseline textual con embeddings multilingues. **Hecho.**
5. Implementar recuperacion visual con encoder preentrenado. **Hecho para long/short.**
6. Implementar fusion multimodal y barrido de `alpha`. **Hecho para long/short.**
7. Correr VLM zero-shot con prompts fijos. **Hecho para longest.**
8. Fine-tunear con LoRA/QLoRA si hay GPU suficiente. **Hecho para longest y enriched.**
9. Evaluar con metricas automaticas y analisis cualitativo. **Metricas hechas; revision manual pendiente.**
10. Consolidar resultados en formato de paper ACL. **Pendiente.**

## Decisiones abiertas

- Modelo VLM concreto para zero-shot y LoRA: elegido para la corrida principal,
  `Qwen/Qwen2.5-VL-3B-Instruct`.
- Encoder visual para recuperacion de imagenes: implementado en scripts de
  retrieval visual/multimodal; revisar solo si se cambia de modelo.
- Encoder textual multilingue: TF-IDF, E5 y SBERT ya implementados.
- Regla definitiva para seleccionar target de referencia.
- Uso o no de metadatos de autor/ranking en filtrado de respuestas.
- Presupuesto de GPU y tamano maximo de modelo: para la corrida enriched se uso
  una VM L4; queda reportar costo final de billing si esta disponible.

## Definicion de exito

El proyecto sera exitoso si produce una comparacion clara, reproducible y
honesta entre recuperacion multimodal y fine-tuning liviano para VQA
dermatologico en espanol, incluyendo fortalezas, limitaciones y condiciones bajo
las cuales cada enfoque conviene.
