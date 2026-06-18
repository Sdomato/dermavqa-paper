# Plan de ejecución — Matías

> VLM (zero-shot y LoRA/QLoRA) sobre `dataset_longest_answer` + liderazgo del paper.
> Derivado de `team_execution_plan.md`, adaptado al estado real del repo (rama `develop`).

## Resumen de mi parte

Soy responsable de la pata **VLM sobre respuesta larga**. Tengo que producir tres
métodos comparables sobre `dataset_longest_answer` y dejar todo medido con el mismo
protocolo que Santino:

1. VLM **zero-shot** (imagen + pregunta → respuesta larga).
2. VLM **LoRA/QLoRA** fine-tuneado sobre `dataset_longest_answer`.
3. Comparar 1 y 2 contra el **retrieval multimodal** de Damián (mismo dataset).

Y además **liderar la escritura del paper**.

La comparación estrella del trabajo es:
**LoRA sobre enriquecido (Santino) vs LoRA sobre respuesta larga (yo)**.

---

## Estado del repo que me condiciona (leer antes de arrancar)

- El dataset canónico es `outputs/datasets/dataset_longest_answer.json` (998 casos).
  Claves por caso:
  `encounter_id`, `image_ids` (lista, puede tener >1), `query_title_es`,
  `query_content_es`, `answer_es` (mi target), `_split`, `n_responses`.
- Splits reales en `_split`: `train` (842), `valid_ht` (56),
  `test_ht_spanishtestsetcorrected` (100). **Todos los casos tienen imágenes.**
- Los baselines de retrieval ya están corridos y medidos por Damián:
  `src/{tfidf,e5,sbert,visual,multimodal}_retrieval.py` +
  `src/evaluate_retrieval.py`. Sus métricas están en
  `outputs/metrics/dataset_longest_answer/metrics_summary.csv`.
- **`src/train.py` de la rama `feature/training-pipeline` NO sirve tal cual:** lee un
  formato viejo (`train.json` crudo vía `prepare_dataset.py`). Hay que reescribirlo
  para que consuma `dataset_longest_answer.json` directamente.
- **Inconsistencia de carpetas a resolver con el equipo:** el plan dice `results/...`
  pero el código de `develop` usa `outputs/results/` (predicciones JSON) y
  `outputs/metrics/` (CSVs de métricas). Decisión propuesta: **seguir lo que ya hace
  el código** (`outputs/results/` + `outputs/metrics/`) y avisar al equipo, para no
  romper `evaluate_retrieval.py`.
- Imágenes: `retrieval_utils.py` espera `data/images`, pero el pipeline de training
  tenía `data/iiyi/images_final` + `images_final.zip`. **Confirmar una sola ruta
  canónica** antes de generar nada (ver Fase 0).
- `evaluate_retrieval.py` importa `rouge_score` y `bert_score`; verificar que estén
  en `requirements.txt` de `develop`.

---

## Fase 0 — Setup y reconciliación (medio día)

- [x] Branch desde `develop` (no desde `feature/training-pipeline`):
      `git checkout develop && git pull && git checkout -b feature/vlm-longest-answer`.
- [x] Confirmar/unificar la ruta de imágenes. Las 998 imágenes resuelven bajo
      `data/iiyi/images_final/`; `vlm_infer.py` las resuelve por `rglob` sobre esa ruta
      (no hizo falta symlink a `data/images`).
- [x] Verificar que cada `image_ids[i]` resuelve a un archivo real: valid 56, test 100,
      train 842, **0 faltantes**.
- [x] Alinear `requirements.txt`: `transformers`, `qwen-vl-utils`, `accelerate`,
      `peft`, `bitsandbytes`, `trl`, `rouge_score`, `bert-score`, `sacrebleu`,
      `sentence-transformers`/E5 para la cosine semántica.
- [x] Definir helper compartido de carga del dataset por split (reusa
      `src/retrieval_utils.py::load_dataset` y `build_query_text`).

---

## Fase 1 — VLM zero-shot (`src/vlm_infer.py`)

Objetivo: predecir la respuesta larga a partir de imagen(es) + pregunta, sin entrenar.

- [x] Script `src/vlm_infer.py` (validado en CPU con `--dry-run`):
  - Carga `Qwen/Qwen2.5-VL-7B-Instruct` (4-bit para que entre en T4/L4).
  - Para cada caso de un split arma el mensaje chat:
    system (prompt de dermatólogo en español) + user con `image_ids` (todas) +
    `build_query_text(record)` como texto.
  - Genera con `do_sample=False` (determinista) y `max_new_tokens` razonable (p.ej. 256).
  - Soporta `--split valid|test` y `--limit N` para pruebas rápidas.
- [x] Salida en CSV con el **esquema canónico del plan de equipo**:
  `split, encounter_id, image_id, question_es, reference_answer_es,
  predicted_answer_es, model_name, dataset_variant, method`
  (con `method=vlm_zero_shot`, `dataset_variant=longest_answer`).
- [ ] Guardar en `outputs/results/dataset_longest_answer/vlm_zero_shot/`:
  `predictions_valid.csv`, `predictions_test.csv`. *(produce archivos en la corrida GPU)*
- [ ] Loguear métricas operativas: tiempo medio de inferencia por ejemplo y VRAM.
  *(el código ya escribe `runtime_<split>.json`; los valores salen en la corrida GPU)*

---

## Fase 2 — Fine-tuning LoRA/QLoRA (`src/train_longest.py`)

Objetivo: fine-tunear el mismo VLM sobre `dataset_longest_answer` (solo `train`).

- [x] Reescribir el training para consumir `dataset_longest_answer.json`
      (`src/train_longest.py`, validado en CPU con `--dry-run`):
  - Filtra `_split == "train"` para entrenar; `valid_ht` para selección de checkpoint.
  - Formatea cada caso al formato chat de Qwen2.5-VL reusando `build_chat_messages`
    de `vlm_infer.py` (imágenes `{"type":"image","image":<path>}` + texto pregunta;
    assistant = `answer_es`). Collator multimodal enmascara prompt e imágenes en la loss.
  - QLoRA 4-bit, `SFTTrainer` (TRL), `LoraConfig` r=16 α=32, con el loader nuevo.
  - Descarta casos sin imagen resuelta o sin `answer_es`.
- [x] Hiperparámetros de arranque: 3 epochs, LR 2e-4, batch 1 + grad_accum 16,
      `eval_steps` sobre valid, `load_best_model_at_end=True` (defaults del CLI).
- [ ] Correr en GCP VM:
  - Actualizar `scripts/setup_vm.sh` → branch `feature/vlm-longest-answer` y el flujo
    de imágenes (GCS o zip).
  - **Apagar la VM al terminar** (`sudo shutdown -h now`).
- [ ] Guardar **solo el adapter** (no el modelo base) en
      `outputs/results/dataset_longest_answer/vlm_lora/final_adapter/`.
- [ ] Métricas operativas: tiempo total de fine-tuning, VRAM pico, **tamaño del adapter**.

---

## Fase 3 — Inferencia con el modelo fine-tuneado

- [x] Extender `src/vlm_infer.py` con `--adapter <path>` para cargar el LoRA sobre la base
      (implementado vía `PeftModel.from_pretrained`, validado con `--dry-run`).
- [ ] Generar predicciones sobre `valid` y `test` con `method=vlm_lora`.
- [ ] Guardar en `outputs/results/dataset_longest_answer/vlm_lora/`:
  `predictions_valid.csv`, `predictions_test.csv` (mismo esquema canónico).

---

## Fase 4 — Evaluación unificada (`src/evaluate_predictions.py`)

Clave: usar **exactamente las mismas métricas** que el equipo para que la comparación
sea justa. `evaluate_retrieval.py` solo lee JSON de retrieval; necesito un evaluador
que lea mis CSVs de predicciones reusando sus funciones de métrica.

- [x] Script `src/evaluate_predictions.py` (validado con predicciones sintéticas)
      que, dado un CSV de predicciones, calcula:
  - **Léxicas:** sacreBLEU, **chrF (principal)**, ROUGE-L, token-F1
    (reusa `token_f1`, `compute_chrf`, `compute_rouge_l` de `evaluate_retrieval.py`).
  - **Semánticas:** BERTScore F1 multilingüe + cosine similarity con embeddings E5.
- [ ] Salida: `metrics_valid.csv` y `metrics_test.csv` por método, en
      `outputs/metrics/dataset_longest_answer/`.
- [ ] Tabla comparativa final sobre `longest_answer`:
      **vlm_zero_shot vs vlm_lora vs retrieval_multimodal (Damián)**.

---

## Fase 5 — Revisión clínica manual (`manual_review_20.csv`)

- [ ] Muestrear ~20 casos (mismos `encounter_id` para los tres métodos).
- [ ] Anotar a mano, por caso y método:
  `diagnosis_supported_rate`, `unsafe_recommendation_rate`, `hallucination_rate`,
  `too_generic_rate`, `empty_or_invalid_rate`.
- [ ] Guardar `manual_review_20.csv` en
      `outputs/results/dataset_longest_answer/`.

---

## Fase 6 — Métricas operativas consolidadas

- [ ] Tabla con: tiempo medio de inferencia/ejemplo (zero-shot vs LoRA), tiempo total
      de fine-tuning, VRAM, tamaño del adapter, costo cloud aproximado.

---

## Fase 7 — Escritura del paper (lidero)

- [ ] Estructura: motivación · datasets (3 variantes) · métodos (retrieval
      textual/visual/multimodal + VLM zero-shot/LoRA) · resultados · limitaciones ·
      **conclusión retrieval vs fine-tuning**.
- [ ] Pedir y consolidar resultados de Santino (LoRA enriquecido) y Damián (retrieval).
- [ ] Tablas: métricas por dataset; **comparación estrella enriquecido-LoRA (Santino)
      vs longest-LoRA (yo)**, cada modelo contra su propio target + cruce opcional.
- [ ] La decisión final NO se basa en una sola métrica: combinar chrF + BERTScore +
      menor hallucination/unsafe + revisión manual + costo/tiempo + claridad clínica.

---

## Progreso de implementación

### Entorno (conda)

```bash
conda create -n dermavqa python=3.11 -y
conda activate dermavqa
pip install -r requirements.txt          # en el box con GPU (torch, transformers, etc.)
# Para validar solo lo léxico en local (sin GPU) alcanza con:
#   pip install pandas numpy sacrebleu rouge_score
```

### Hecho (CPU, sin GPU)

- [x] Rama `feature/vlm-longest-answer` creada desde `origin/develop`.
- [x] Confirmado que las 998 imágenes resuelven en local
      (`data/iiyi/images_final/`): valid 56, test 100, train 842, **0 faltantes**.
- [x] `src/vlm_infer.py` — inferencia VLM zero-shot + soporte `--adapter` (LoRA).
      Carga de dataset, armado de prompt multimodal y resolución de imágenes
      validados con `--dry-run`. Falta solo la generación real (GPU).
- [x] `src/evaluate_predictions.py` — evaluador con las MISMAS métricas que
      `evaluate_retrieval.py` (chrF, ROUGE-L, token-F1, sacreBLEU corpus;
      BERTScore y cosine E5 opcionales). Validado con predicciones sintéticas:
      perfecta → chrF/ROUGE-L = 1.0, sacreBLEU = 100; mala → ~0.
- [x] `requirements.txt`: agregadas deps de VLM fine-tuning
      (accelerate, peft, bitsandbytes, trl, qwen-vl-utils).
- [x] `src/train_longest.py` — fine-tuning QLoRA (Fase 2). Loader sobre
      `dataset_longest_answer.json` (train + valid_ht), formato chat reusado de
      `vlm_infer.py`, collator multimodal con masking de prompt/imágenes,
      `SFTTrainer` + `LoraConfig` r=16 α=32. Validado con `--dry-run` (resuelve
      casos multi-imagen, arma system+user+assistant). Falta solo correr en GPU.

Nota: `token_f1` usa `set()` de tokens igual que `evaluate_retrieval.py`, por eso
un match perfecto da ~0.83 y no 1.0. Es la definición compartida del equipo; se
mantiene a propósito para que VLM y retrieval sean comparables.

### Pendiente de GPU (una sola sesión)

- [ ] `python -m src.vlm_infer --split valid` y `--split test`  (zero-shot real).
- [ ] `python -m src.train_longest`  (fine-tuning QLoRA real; el script ya está escrito).
- [ ] `python -m src.vlm_infer --split test --adapter <ruta>`  (inferencia LoRA).
- [ ] `python -m src.evaluate_predictions <preds...>`  (con BERTScore/cosine).

### Cómo correr (referencia rápida)

```bash
# 1. Validar prompts e imágenes sin modelo (local, CPU)
python -m src.vlm_infer --split valid --limit 5 --dry-run

# 2. Zero-shot (GPU)
python -m src.vlm_infer --split valid
python -m src.vlm_infer --split test

# 3. Fine-tuning QLoRA (GPU) — valida el formato antes con --dry-run
python -m src.train_longest --dry-run --limit 5        # CPU, chequea chat e imágenes
python -m src.train_longest                            # entrena 3 epochs, guarda adapter
python -m src.vlm_infer --split test \
    --adapter outputs/results/dataset_longest_answer/vlm_lora/final_adapter

# 4. Evaluar (sin GPU si se omite BERTScore)
python -m src.evaluate_predictions \
    outputs/results/dataset_longest_answer/vlm_zero_shot/predictions_test.csv \
    --no-bertscore
```

### Nota sobre rutas y .gitignore

- Predicciones → `outputs/results/dataset_longest_answer/<method>/` (gitignored, local).
- Métricas resumidas → `outputs/metrics/dataset_longest_answer/` (sí se versiona).
- Imágenes nunca se versionan (`data/iiyi/images_final/` está en `.gitignore`).

## Reglas a respetar (del plan de equipo)

- Entrenar solo con `train`; `valid` para elegir checkpoint; `test` solo para el reporte final.
- **No subir** imágenes, claves, `.env`, checkpoints pesados ni caches. El adapter LoRA
  va aparte / documentado si pesa.
- Subir notebooks, scripts, métricas resumidas y docs del survey.
- Nombres de carpetas consistentes por dataset y método.

## Entregables míos

- `src/vlm_infer.py` (zero-shot + `--adapter`), `src/train_longest.py`,
  `src/evaluate_predictions.py`.
- `outputs/results/dataset_longest_answer/vlm_zero_shot/` y `.../vlm_lora/`.
- `outputs/metrics/dataset_longest_answer/metrics_{valid,test}.csv`.
- `manual_review_20.csv` + tabla operativa.
- Borrador del paper con tablas y la conclusión retrieval vs fine-tuning.
