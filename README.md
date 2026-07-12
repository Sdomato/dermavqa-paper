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

## Líneas de trabajo

| Línea | Foco | Dataset principal |
| --- | --- | --- |
| Dataset enriquecido | Dataset enriquecido (síntesis LLM) + LoRA/QLoRA sobre enriquecido | `dataset_enriched` |
| Retrieval | Construcción de datasets + baselines de retrieval (textual, visual, multimodal) | `dataset_longest_answer_by_image`, `dataset_enriched` |
| VLM respuesta larga | VLM zero-shot + LoRA/QLoRA sobre respuesta larga | `dataset_longest_answer_by_image` |

**Comparación estrella del trabajo:** LoRA sobre respuesta larga vs LoRA
sobre enriquecido, ambos contra retrieval multimodal.

## Estructura del repo

```text
.
├── data/iiyi/                 # Subconjunto IIYI: casos, metadatos e imágenes (imágenes NO versionadas)
│   ├── train.json             # 842 casos
│   ├── valid_ht.json          # 56 casos
│   ├── test_ht_spanishtestsetcorrected.json  # 100 casos (test ES canónico)
│   └── images_final/          # 2.945 imágenes (copiar manualmente tras clonar — ver Datos)
├── notebooks/                 # Exploración, baselines, VLM enriched y resultados paper-ready
├── paper/                     # Borrador narrativo del paper
├── src/                        # Pipeline modular (ver Scripts)
├── outputs/                   # Raíz canónica de artefactos (ver survey/STRUCTURE.md)
│   ├── datasets/              # Datasets procesados (versionados): longest_by_image / enriched.zip
│   ├── results/<dataset>/<método>/   # Predicciones + artefactos por método (CSV livianos versionados; .npy/adapters NO)
│   ├── metrics/<dataset>/     # Métricas resumidas por dataset (versionado)
│   └── paper/{tables,figures}/ # Tablas y figuras finales para el paper (versionado)
├── survey/                    # Survey, planes de ejecución, notas metodológicas y STRUCTURE.md
├── config.yaml.example        # Plantilla de config (copiar a config.yaml)
└── requirements.txt
```

> **Convención de carpetas (unificada):** la raíz canónica de resultados es
> `outputs/` — el código escribe en `outputs/results/` (predicciones) y
> `outputs/metrics/` (métricas) y `outputs/paper/` (tablas/figuras finales).
> El borrador narrativo vive en `paper/draft.md`.
> El antiguo `results/` de la raíz quedó
> deprecado y su contenido (enriched) se migró a
> `outputs/metrics/dataset_enriched/`. Layout completo en `survey/STRUCTURE.md`.

## Datasets (variantes de target)

Cada caso tiene una o más imágenes, una consulta del paciente en español y
**varias** respuestas médicas. Como no hay un único target obvio, se preparan
varias vistas del mismo corpus:

| Variante | Target | Archivo / artefacto |
| --- | --- | --- |
| `longest_answer_by_image` | Respuesta original más larga, una fila por imagen (target canónico de respuesta larga: retrieval, zero-shot y LoRA se evalúan todos con esta misma unidad, ver "Nota de versión" en `paper/draft.md` §5) | `outputs/datasets/dataset_longest_answer_by_image.{json,jsonl,csv,zip}` |
| `enriched` | Respuestas consolidadas por un LLM (Azure), una fila por imagen | `outputs/datasets/dermavqa_iiyi_llm_synthesized_answer_finetune.zip` |

Ambas variantes comparten esquema (una fila por imagen) y por eso son las
únicas evaluadas por `evaluate_retrieval_heldout.py`: retrieval, zero-shot y
LoRA siempre se comparan sobre la misma unidad. `dataset_short_answer` se
eliminó del repo (código y outputs) — dejó de ser un target de interés.

`dataset_longest_answer.{json,csv}` (una fila por caso, sin expandir a
imagen) es un artefacto **intermedio**: `build_answer_datasets.py` lo genera
primero y `build_longest_by_image_dataset.py` lo expande a
`dataset_longest_answer_by_image`. No se usa como target final de ningún
resultado reportado — la corrida de VLM sobre una imagen por caso quedó
superada porque no era comparable con los baselines de retrieval y el
zero-shot, evaluados por imagen (ver `paper/draft.md` §5).

Esquema del dataset canónico (`dataset_longest_answer_by_image.json`, una
fila por imagen): `split`, `encounter_id`, `image_id`, `image_path`,
`question_es`, `answer_es`. Splits: `train` (842 casos), `valid_ht` (56
casos, 157 imágenes), `test_ht_spanishtestsetcorrected` (100 casos, 314
imágenes).

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
| `build_answer_datasets.py` | Genera `dataset_longest_answer` (intermedio, una fila por caso) desde los JSON crudos de IIYI. |
| `build_longest_by_image_dataset.py` | Expande `dataset_longest_answer` a `dataset_longest_answer_by_image` (una fila por imagen), el target canónico de respuesta larga usado en todos los resultados reportados. |
| `build_llm_synthesized_dataset.py` | Genera el dataset enriquecido vía Azure OpenAI (síntesis extractiva, con trazabilidad). Ver `survey/dataset_notes.md`. |

### Baselines de retrieval
| Script | Modelo / método |
| --- | --- |
| `tfidf_retrieval.py` | TF-IDF (sklearn) |
| `e5_retrieval.py` | Multilingual E5 (`intfloat/multilingual-e5-base`) |
| `sbert_retrieval.py` | Sentence-BERT (`paraphrase-multilingual-MiniLM-L12-v2`) |
| `retrieval_utils.py` | Utilidades compartidas (carga de dataset, query text, top-1). |
| `evaluate_retrieval.py` | Métricas de los baselines: `--dataset longest_answer`. |
| `plot_retrieval_scores.py` | Gráficos de distribución de scores (PNG 300 dpi). |

> Los scripts de arriba corren top-1 sobre todo el corpus (`dataset_longest_answer`,
> una fila por caso) y sirven para exploración rápida en CPU. **Los números
> reportados en el paper vienen de `evaluate_retrieval_heldout.py`**, que
> evalúa train-only (sin data leakage) sobre `dataset_longest_answer_by_image`
> y `dataset_enriched` — la misma unidad que usan zero-shot y LoRA, para que
> todas las filas de la Tabla 1 sean comparables entre sí. Incluye textual
> (TF-IDF/E5/SBERT), visual (BiomedCLIP) y multimodal (E5+BiomedCLIP late
> fusion, `--alpha`, default 0.6). visual/multimodal reemplazan a los viejos
> `visual_retrieval.py`/`multimodal_retrieval.py` (eliminados): esos corrían
> sobre todo el corpus mezclado — filtraban valid/test al índice de
> recuperación — y sobre `dataset_longest_answer` en vez de `_by_image`.
> visual/multimodal requieren imágenes locales (`data/iiyi/images_final/`) y
> `open_clip_torch`:
> ```bash
> python -m src.evaluate_retrieval_heldout --dataset dataset_longest_answer_by_image --methods tfidf,e5,sbert,visual,multimodal
> python -m src.evaluate_retrieval_heldout --dataset dataset_enriched --methods tfidf,e5,sbert,visual,multimodal
> ```
> Corré todos los métodos que quieras reportar **en una sola invocación**:
> cada corrida sobreescribe `metrics_summary.csv`/`metrics_per_case.csv`
> completos (no los mergea con corridas anteriores).

### VLM
| Script | Qué hace |
| --- | --- |
| `train_longest_by_image.py` | Fine-tuning QLoRA sobre `dataset_longest_answer_by_image` (todas las imágenes por caso, early stopping), con el mismo motor que enriched. **Script canónico de respuesta larga.** |
| `vlm_infer_longest_by_image.py` | Inferencia Qwen2.5-VL sobre `dataset_longest_answer_by_image` (zero-shot o con `--adapter` LoRA). **Script canónico de respuesta larga.** |
| `train_enriched.py` | Fine-tuning QLoRA by-image sobre `dataset_enriched`; wrapper de `vlm_lora_training.py`. |
| `vlm_infer_enriched.py` | Inferencia Qwen2.5-VL sobre `dataset_enriched`, compatible con adapter LoRA; wrapper de `vlm_by_image_utils.py`. |
| `vlm_by_image_utils.py` | Utilidades compartidas para datasets by-image: loader, resolución de imágenes, prompt, modelo e inferencia. |
| `vlm_lora_training.py` | Motor compartido de QLoRA reproducible: seed, collator, LoRA config, runtime, métricas y logs. |
| `evaluate_predictions.py` | Mismas métricas que `evaluate_retrieval.py` sobre los CSV de predicciones del VLM (comparabilidad). |
| `build_paper_results.py` | Consolida métricas y genera tablas/figuras SVG paper-ready en `outputs/paper/`. |

> El fine-tuning/inferencia con una imagen por caso (sin expandir a
> `by_image`) quedó **superado**: esa corrida no era comparable con los
> baselines de retrieval ni con el zero-shot, evaluados por imagen. Los
> scripts correspondientes se eliminaron; el log de esa corrida queda
> documentado únicamente en `paper/draft.md` §5.

## How to Reproduce

Todos los comandos se ejecutan desde la raíz del repositorio. Las semillas están fijadas en `42` en todos los scripts de entrenamiento.

### Árbol de scripts → artefactos

| Script | Artefacto generado |
| --- | --- |
| `src/build_answer_datasets.py` | `outputs/datasets/dataset_longest_answer.{json,csv}` (intermedio) |
| `src/build_longest_by_image_dataset.py` | `outputs/datasets/dataset_longest_answer_by_image.*` (target canónico de respuesta larga) |
| `src/build_llm_synthesized_dataset.py` | `outputs/datasets/dermavqa_iiyi_llm_synthesized_answer_finetune.*` (requiere Azure key) |
| `src/evaluate_retrieval_heldout.py --dataset dataset_longest_answer_by_image\|dataset_enriched --methods tfidf,e5,sbert,visual,multimodal` | `outputs/metrics/<dataset_variant>/retrieval_heldout/metrics_{summary,per_case}.csv` — **retrieval reportado en la Tabla 1** (train-only, sin leakage) |
| `src/tfidf_retrieval.py` / `sbert_retrieval.py` / `e5_retrieval.py` | `outputs/results/dataset_longest_answer/retrieval_textual*/*` (exploración CPU, no reportado en Tabla 1) |
| `src/build_paper_results.py` | `outputs/paper/{tables,figures}/*` |
| `src/train_longest_by_image.py` / `scripts/run_longest_by_image_vlm_lora.sh` | `outputs/results/dataset_longest_answer/vlm_lora_by_image/` — **LoRA reportado en la Tabla 1** |
| `src/vlm_infer_longest_by_image.py` | `outputs/results/dataset_longest_answer/vlm_zero_shot_by_image/predictions_{valid,test}.csv` — **zero-shot reportado en la Tabla 1** |
| `scripts/run_enriched_vlm_lora.sh` | `outputs/results/dataset_enriched/vlm_lora/` |
| `scripts/run_vlm_rag_comparison.sh` | `outputs/results/dataset_*/vlm_*_rag_*/predictions_*.csv` |

---

### Fase 1 — CPU (sin GPU)

```bash
# Setup
conda create -n dermavqa python=3.11 -y && conda activate dermavqa
pip install -r requirements.txt
cp config.yaml.example config.yaml

# Copiar imágenes manualmente (~1.2 GB) a:
#   data/iiyi/images_final/{images_train,images_valid,images_test}/

# Un comando para correr todo lo que no requiere GPU:
make all
```

`make all` ejecuta en orden: construcción de datasets → retrieval baselines → métricas de retrieval → tablas y figuras paper-ready.

Para ver todos los targets disponibles:

```bash
make help
```

Para validar el pipeline sin cargar ningún modelo (CPU puro):

```bash
make dry-run
```

---

### Fase 2 — GPU: fine-tuning y VLM

```bash
# Validar formato de datos sin GPU (recomendado antes de lanzar en la nube)
python -m src.vlm_infer_longest_by_image --split valid --limit 5 --dry-run
python -m src.train_longest_by_image --dry-run --limit 5

# LoRA sobre dataset_longest_answer_by_image (todas las imágenes por caso, early stopping)
bash scripts/run_longest_by_image_vlm_lora.sh --seed 42

# LoRA sobre dataset_enriched (respuestas sintetizadas por LLM)
bash scripts/run_enriched_vlm_lora.sh --seed 42

# VLM zero-shot sobre dataset_longest_answer_by_image (sin fine-tuning)
python -m src.vlm_infer_longest_by_image --split valid
python -m src.vlm_infer_longest_by_image --split test
```

---

### Fase 3 — GPU: experimentos RAG

```bash
# Zero-shot y LoRA con recuperación de contexto (requiere adapters ya entrenados)
bash scripts/run_vlm_rag_comparison.sh
```

---

### Fase 4 — CPU: evaluación final y paper

```bash
# Evaluar predicciones VLM generadas
python -m src.evaluate_predictions \
    outputs/results/dataset_longest_answer/vlm_zero_shot_by_image/predictions_test.csv

# Regenerar tablas y figuras paper-ready con todos los resultados disponibles
make paper
```

---

### Notas de reproducibilidad

- **Seeds:** todos los scripts de entrenamiento usan `--seed 42` por defecto (`random`, `numpy`, `torch`, `transformers.set_seed`, `SFTConfig.seed`).
- **Retrieval:** TF-IDF, SBERT y E5 son determinísticos dado el dataset. BiomedCLIP puede tener variación mínima en GPU multi-thread; para reproducibilidad exacta usar CPU.
- **dataset_enriched:** la síntesis LLM vía Azure usa temperatura 0.2 y está cacheada en `outputs/datasets/llm_synthesis/cache/`. Re-sintetizar requiere las mismas credenciales Azure y puede dar resultados ligeramente distintos.

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
- Datasets `longest_by_image` / `enriched` construidos y versionados.
- Baselines de retrieval held-out textuales (TF-IDF, E5, SBERT; train-only,
  sin leakage) corridos sobre `dataset_longest_answer_by_image`.
- Baselines held-out visual (BiomedCLIP) y multimodal (E5+BiomedCLIP)
  agregados a `evaluate_retrieval_heldout.py` y corridos sobre
  `dataset_longest_answer_by_image` y `dataset_enriched`.
- Retrieval textual enriquecido corrido (en `outputs/metrics/dataset_enriched/`).
- VLM zero-shot y LoRA sobre `dataset_longest_answer_by_image` corridos con
  Qwen2.5-VL-3B (todas las imágenes por caso, early stopping en L4).
- LoRA/QLoRA sobre `dataset_enriched` corrido en Google Cloud L4 por 1 epoch.
- Predicciones y métricas de `dataset_enriched/vlm_lora` guardadas en
  `outputs/results/dataset_enriched/vlm_lora/` y
  `outputs/metrics/dataset_enriched/metrics_mixed.csv`.

**Falta:** revisión clínica manual de ~20 casos, análisis cruzado entre
targets si se decide hacerlo, tabla de costos y escritura final del paper.

### Resultados de respuesta larga (dataset_longest_answer_by_image, test n=100)

Los cinco métodos evaluados con la misma unidad (por imagen, agregado por
caso), sin data leakage — ver `paper/draft.md` §5, Tabla 1:

| Método | sacreBLEU | chrF corpus | chrF mean | ROUGE-L | token-F1 | BERTScore-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-IDF held-out | 0.785 | 13.501 | 0.156 | 0.057 | 0.075 | – |
| E5 held-out | 0.305 | 13.956 | 0.156 | 0.076 | 0.096 | – |
| SBERT held-out | 0.633 | 16.735 | 0.169 | 0.073 | 0.097 | – |
| Visual (BiomedCLIP) held-out | 0.433 | 14.872 | 0.173 | 0.076 | 0.102 | – |
| Multimodal (E5+BiomedCLIP) held-out | 0.323 | 12.819 | 0.153 | 0.066 | 0.085 | – |
| **Qwen2.5-VL zero-shot** | 0.352 | 16.163 | **0.204** | 0.080 | 0.103 | 0.643 |
| Qwen2.5-VL LoRA | 0.211 | 13.810 | 0.161 | 0.080 | 0.088 | 0.618 |

Zero-shot rinde mejor que LoRA y que los cinco retrieval en todas las métricas
salvo sacreBLEU. Entre los retrieval, SBERT y visual quedan prácticamente
empatados en chrF; la fusión multimodal no mejora sobre su mejor componente
individual. LoRA no mejora sobre zero-shot en respuesta larga — a diferencia
del dataset enriquecido, donde LoRA sí supera a retrieval (ver tabla
siguiente). El paper ACL (`paper/acl/main.tex`, Tabla 2) compara los
recuperadores entre sí sin contrastarlos contra las condiciones generativas;
detalle e interpretación de la versión más larga en `paper/draft.md` §5
(desactualizado respecto al ACL, ver nota en el repo).

(fuente: `outputs/metrics/dataset_longest_answer_by_image/retrieval_heldout/metrics_summary.csv`
y `outputs/paper/tables/paper_main_test_comparison.csv`)

### Resultados VLM enriquecido disponibles (dataset_enriched)

| Split | n por imagen | chrF mean | ROUGE-L | token-F1 | sacreBLEU | BERTScore-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| valid | 157 | 0.361 | 0.243 | 0.305 | 10.904 | 0.737 |
| test | 314 | 0.365 | 0.254 | 0.317 | 11.598 | 0.738 |

(fuente: `outputs/metrics/dataset_enriched/metrics_mixed.csv`)

## Convenciones de colaboración

- **No** subir imágenes, claves, `.env`, checkpoints pesados ni caches.
- Subir notebooks, scripts, métricas resumidas y docs del survey.
- Subir tablas/figuras finales de paper bajo `outputs/paper/`.
- Mantener nombres de carpetas consistentes por dataset y método.
- El adapter LoRA va documentado aparte si pesa.
