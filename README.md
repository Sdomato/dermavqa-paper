# DermaVQA-IIYI — Retrieval vs Fine-tuning para VQA dermatológico en español

Estudio empírico sobre **Visual Question Answering dermatológico en español**
usando el subconjunto IIYI de DermaVQA. La pregunta central es metodológica:

> Bajo restricciones realistas de datos y cómputo, ¿conviene **adaptar** un VLM
> con LoRA/QLoRA o **recuperar** la respuesta de casos dermatológicos similares
> (retrieval multimodal)?

El objetivo **no** es una herramienta clínica desplegable, sino una comparación
reproducible y honesta entre estrategias de adaptación para un dominio médico
multimodal y subrepresentado en español. Ver `survey/README.md` para el roadmap
completo y `survey/team_execution_plan.md` para la división de tareas.

## Equipo y responsabilidades

| Persona | Foco | Dataset principal |
| --- | --- | --- |
| **Santino** | Dataset enriquecido (síntesis LLM) + LoRA/QLoRA sobre enriquecido | `dataset_enriched` |
| **Damián** | Construcción de datasets long/short + baselines de retrieval (textual, visual, multimodal) | `dataset_longest_answer`, `dataset_short_answer` |
| **Matías** | VLM zero-shot + LoRA/QLoRA sobre respuesta larga + liderazgo del paper | `dataset_longest_answer` |

**Comparación estrella del trabajo:** LoRA sobre respuesta larga (Matías) vs
LoRA sobre enriquecido (Santino), ambos contra retrieval multimodal (Damián).

## Estructura del repo

```text
.
├── data/iiyi/                 # Subconjunto IIYI: casos, metadatos e imágenes (imágenes NO versionadas)
│   ├── train.json             # 842 casos
│   ├── valid_ht.json          # 56 casos
│   ├── test_ht_spanishtestsetcorrected.json  # 100 casos (test ES canónico)
│   └── images_final/          # 2.945 imágenes (copiar manualmente tras clonar — ver Datos)
├── notebooks/                 # Exploración (01) y baseline textual (02)
├── src/                        # Pipeline modular (ver Scripts)
├── outputs/                   # Raíz canónica de artefactos (ver survey/STRUCTURE.md)
│   ├── datasets/              # Datasets procesados (versionados): long / short / enriched.zip
│   ├── results/<dataset>/<método>/   # Predicciones + artefactos por método (CSV livianos versionados; .npy/adapters NO)
│   └── metrics/<dataset>/     # Métricas resumidas por dataset (versionado)
├── survey/                    # Survey, planes de ejecución, notas metodológicas y STRUCTURE.md
├── config.yaml.example        # Plantilla de config (copiar a config.yaml)
└── requirements.txt
```

> **Convención de carpetas (unificada):** la raíz canónica de resultados es
> `outputs/` — el código escribe en `outputs/results/` (predicciones) y
> `outputs/metrics/` (métricas). El antiguo `results/` de la raíz quedó
> deprecado y su contenido (enriched) se migró a
> `outputs/metrics/dataset_enriched/`. Layout completo en `survey/STRUCTURE.md`.

## Datasets (variantes de target)

Cada caso tiene una o más imágenes, una consulta del paciente en español y
**varias** respuestas médicas. Como no hay un único target obvio, se preparan
varias vistas del mismo corpus:

| Variante | Target | Archivo / artefacto |
| --- | --- | --- |
| `longest_answer` | Respuesta original más larga | `outputs/datasets/dataset_longest_answer.{json,csv}` |
| `short_answer` | Respuesta más corta (diagnóstico breve) | `outputs/datasets/dataset_short_answer.{json,csv}` |
| `enriched` | Respuestas consolidadas por un LLM (Azure) | `outputs/datasets/dermavqa_iiyi_llm_synthesized_answer_finetune.zip` |

Esquema del dataset canónico (`dataset_longest_answer.json`): `encounter_id`,
`author_id`, `image_ids` (lista), `query_title_es`, `query_content_es`,
`query_title_en`, `query_content_en`, `_split`, `answer_es`, `answer_en`,
`answer_author_id`, `n_responses`. Splits en `_split`: `train` (842),
`valid_ht` (56), `test_ht_spanishtestsetcorrected` (100).

## Setup

```bash
conda create -n dermavqa python=3.11 -y
conda activate dermavqa
pip install -r requirements.txt

cp config.yaml.example config.yaml   # editar solo si las imágenes están en otro path
```

Para validar **solo métricas léxicas en local sin GPU** alcanza con un subconjunto:

```bash
pip install pandas numpy sacrebleu rouge_score
```

### Datos e imágenes

Las imágenes (~1.2 GB) **no se versionan**. Tras clonar, copiá manualmente el
contenido a `data/iiyi/images_final/` (subcarpetas `images_train/`,
`images_valid/`, `images_test/`). Las 998 imágenes referenciadas por los casos
resuelven en local (0 faltantes).

## Scripts (`src/`)

Todos se ejecutan como módulo desde la raíz del repo (`python -m src.<nombre>`).

### Construcción de datasets
| Script | Qué hace |
| --- | --- |
| `build_answer_datasets.py` | Genera `dataset_longest_answer` y `dataset_short_answer` desde los JSON crudos de IIYI. |
| `build_llm_synthesized_dataset.py` | Genera el dataset enriquecido vía Azure OpenAI (síntesis extractiva, con trazabilidad). Ver `survey/dataset_notes.md`. |

### Baselines de retrieval (Damián)
Cada modalidad tiene dos variantes: `<x>_retrieval.py` (longest) y `<x>_retrieval_short.py` (short).
| Script | Modelo / método |
| --- | --- |
| `tfidf_retrieval.py` | TF-IDF (sklearn) |
| `e5_retrieval.py` | Multilingual E5 (`intfloat/multilingual-e5-base`) |
| `sbert_retrieval.py` | Sentence-BERT (`paraphrase-multilingual-MiniLM-L12-v2`) |
| `visual_retrieval.py` | BiomedCLIP (solo imagen) |
| `multimodal_retrieval.py` | Late fusion texto+imagen: `s = α·s_text + (1-α)·s_image` |
| `retrieval_utils.py` | Utilidades compartidas (carga de dataset, query text, top-1). |
| `evaluate_retrieval.py` | Métricas de los baselines: `--dataset longest_answer\|short_answer`. |
| `plot_retrieval_scores.py` | Gráficos de distribución de scores (PNG 300 dpi). |

### VLM (Matías)
| Script | Qué hace |
| --- | --- |
| `vlm_infer.py` | Inferencia Qwen2.5-VL-3B 4-bit zero-shot; `--adapter <path>` para LoRA. `--dry-run` valida prompts/imágenes sin GPU. |
| `train_longest.py` | Fine-tuning QLoRA (r=16, α=32) sobre `train`, selección con `valid`. `--dry-run` valida el formato chat sin GPU. |
| `train_enriched.py` | Mismo fine-tuning QLoRA que `train_longest.py`, pero sobre `dataset_enriched`. |
| `vlm_infer_enriched.py` | Inferencia Qwen2.5-VL sobre `dataset_enriched`, compatible con adapter LoRA. |
| `evaluate_predictions.py` | Mismas métricas que `evaluate_retrieval.py` sobre los CSV de predicciones del VLM (comparabilidad). |

## Cómo correr el pipeline

```bash
# 1. Construir datasets (CPU)
python -m src.build_answer_datasets

# 2. Baselines de retrieval (GPU recomendada para E5/SBERT/visual/multimodal)
python -m src.tfidf_retrieval
python -m src.e5_retrieval
python -m src.sbert_retrieval
python -m src.visual_retrieval
python -m src.multimodal_retrieval --alpha 0.6
python -m src.evaluate_retrieval --dataset longest_answer

# 3. VLM zero-shot (validar sin GPU primero)
python -m src.vlm_infer --split valid --limit 5 --dry-run
python -m src.vlm_infer --split valid
python -m src.vlm_infer --split test

# 4. Fine-tuning QLoRA (GPU)
python -m src.train_longest --dry-run --limit 5
python -m src.train_longest
python -m src.vlm_infer --split test \
    --adapter outputs/results/dataset_longest_answer/vlm_lora/final_adapter

# 4b. Fine-tuning QLoRA equivalente sobre dataset_enriched (Santino)
python -m src.train_enriched --dry-run --limit 5
python -m src.train_enriched
# O en una sola corrida: entrenamiento + valid/test + metricas
bash scripts/run_enriched_vlm_lora.sh --epochs 1
python -m src.vlm_infer_enriched --split valid \
    --adapter outputs/results/dataset_enriched/vlm_lora/final_adapter
python -m src.vlm_infer_enriched --split test \
    --adapter outputs/results/dataset_enriched/vlm_lora/final_adapter

# 5. Evaluación unificada (sin GPU si se omite BERTScore)
python -m src.evaluate_predictions \
    outputs/results/dataset_longest_answer/vlm_zero_shot/predictions_test.csv \
    --no-bertscore
```

## Protocolo de evaluación (común al equipo)

- Entrenar **solo** con `train`; elegir checkpoint con `valid`; reportar con `test`.
- Predicciones en CSV con esquema canónico: `split, encounter_id, image_id,
  question_es, reference_answer_es, predicted_answer_es, model_name,
  dataset_variant, method`.
- Métricas: **léxicas** (chrF principal, ROUGE-L, token-F1, sacreBLEU) +
  **semánticas** (BERTScore F1 multilingüe, cosine E5) + **clínicas manuales**
  (~20 casos: hallucination/unsafe/generic rate) + **operativas** (tiempo, VRAM,
  tamaño adapter, costo). Detalle en `survey/evaluation_plan.md`.

## Estado actual

**Hecho:**
- Datasets `longest` / `short` / `enriched` construidos y versionados.
- Baselines de retrieval textual corridos (TF-IDF, E5, SBERT) sobre long y short.
- Retrieval textual enriquecido corrido (en `outputs/metrics/dataset_enriched/`).
- VLM zero-shot y LoRA sobre `dataset_longest_answer` corridos con Qwen2.5-VL-3B.
- LoRA/QLoRA sobre `dataset_enriched` corrido en Google Cloud L4 por 1 epoch.
- Predicciones y métricas de `dataset_enriched/vlm_lora` guardadas en
  `outputs/results/dataset_enriched/vlm_lora/` y
  `outputs/metrics/dataset_enriched/metrics_mixed.csv`.

**Falta:** revisión clínica manual de ~20 casos, tabla comparativa final
normalizada, análisis cruzado entre targets si se decide hacerlo, tabla de
costos y escritura del paper.

### Resultados de retrieval textual disponibles (longest_answer, n=998)

| Modelo | chrF | ROUGE-L | token-F1 | BERTScore-F1 |
| --- | --- | --- | --- | --- |
| TF-IDF | 0.158 | 0.084 | 0.083 | 0.656 |
| E5 | 0.167 | 0.084 | 0.081 | 0.656 |
| SBERT | 0.174 | 0.088 | 0.085 | 0.657 |

(fuente: `outputs/metrics/dataset_longest_answer/metrics_summary.csv`)

### Resultados VLM enriquecido disponibles (dataset_enriched)

| Split | n por imagen | chrF mean | ROUGE-L | token-F1 | sacreBLEU | BERTScore-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| valid | 157 | 0.361 | 0.243 | 0.305 | 10.904 | 0.737 |
| test | 314 | 0.365 | 0.254 | 0.317 | 11.598 | 0.738 |

(fuente: `outputs/metrics/dataset_enriched/metrics_mixed.csv`)

## Convenciones de colaboración

- **No** subir imágenes, claves, `.env`, checkpoints pesados ni caches.
- Subir notebooks, scripts, métricas resumidas y docs del survey.
- Mantener nombres de carpetas consistentes por dataset y método.
- El adapter LoRA va documentado aparte si pesa.
