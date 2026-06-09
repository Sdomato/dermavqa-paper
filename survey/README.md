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
- `team_execution_plan.md`: division de tareas, protocolo comun y metricas
  para comparar modelos fine-tuned.
- `related_work_matrix.csv`: matriz inicial para organizar papers relacionados.
- `risks.md`: riesgos metodologicos, medicos y operativos.

## Contexto del repo

- `data/iiyi/` contiene el subconjunto IIYI con casos, imagenes y metadatos.
- `notebooks/01_explore.ipynb` ya explora el dataset y construye pares
  imagen-pregunta-respuesta para entrenamiento.
- `requirements.txt` incluye dependencias para EDA, fine-tuning de VLMs y
  evaluacion (`transformers`, `peft`, `accelerate`, `bert-score`, `sacrebleu`).
- `src/` existe pero todavia no contiene la implementacion modular.
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

1. Pasar el notebook exploratorio a scripts reproducibles.
2. Auditar splits, conteos y targets del dataset.
3. Implementar baseline textual TF-IDF.
4. Implementar baseline textual con embeddings multilingues.
5. Implementar recuperacion visual con encoder preentrenado.
6. Implementar fusion multimodal y barrido de `alpha`.
7. Correr VLM zero-shot con prompts fijos.
8. Fine-tunear con LoRA/QLoRA si hay GPU suficiente.
9. Evaluar con metricas automaticas y analisis cualitativo.
10. Consolidar resultados en formato de paper ACL.

## Decisiones abiertas

- Modelo VLM concreto para zero-shot y LoRA.
- Encoder visual para recuperacion de imagenes.
- Encoder textual multilingue.
- Regla definitiva para seleccionar target de referencia.
- Uso o no de metadatos de autor/ranking en filtrado de respuestas.
- Presupuesto de GPU y tamano maximo de modelo.

## Definicion de exito

El proyecto sera exitoso si produce una comparacion clara, reproducible y
honesta entre recuperacion multimodal y fine-tuning liviano para VQA
dermatologico en espanol, incluyendo fortalezas, limitaciones y condiciones bajo
las cuales cada enfoque conviene.
