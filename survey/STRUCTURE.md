# Estructura del repo (referencia canónica)

Este documento fija la convención de carpetas y de artefactos del proyecto.
Si algo en el código o en otro doc contradice esto, esto manda — y hay que
abrir un PR para reconciliarlo.

## Árbol canónico

```text
.
├── data/iiyi/                       # Subconjunto IIYI (imágenes NO versionadas)
│   ├── train.json valid_ht.json test_ht*.json   # casos crudos por split
│   ├── df_*.csv  *.json             # metadatos y mapas de IDs
│   └── images_final/                # imágenes en subdirs images_{train,valid,test}/
├── notebooks/                       # 01_explore, 02_text_retrieval_baseline
├── src/                             # pipeline modular (ver README → Scripts)
├── outputs/                         # ← RAÍZ CANÓNICA de artefactos generados
│   ├── datasets/                    # datasets procesados (VERSIONADO)
│   │   ├── dataset_longest_answer.{json,csv}
│   │   ├── dataset_short_answer.{json,csv}
│   │   └── dermavqa_iiyi_llm_synthesized_answer_finetune.zip   # enriched
│   ├── results/<dataset>/<método>/  # predicciones + artefactos por método
│   └── metrics/<dataset>/           # métricas resumidas (VERSIONADO)
├── survey/                          # survey, planes, notas, este STRUCTURE.md
├── README.md  requirements.txt  config.yaml.example  .gitignore
```

`<dataset>` ∈ `{dataset_longest_answer, dataset_short_answer, dataset_enriched}`
`<método>` ∈ `{retrieval_textual_tfidf, retrieval_textual_e5, retrieval_textual_sbert,
retrieval_visual, retrieval_multimodal, vlm_zero_shot, vlm_lora}`

## Qué va en cada carpeta

| Carpeta | Contenido | ¿Versionado en git? |
| --- | --- | --- |
| `outputs/datasets/` | Datasets procesados | Sí: `*.json`, `*.csv`, `*.zip` |
| `outputs/metrics/<dataset>/` | Resúmenes de métricas (`metrics_summary.csv`, `bertscore_summary.csv`, `metrics_{valid,test}.csv`, `manual_review_*.csv`) | Sí (CSV livianos) |
| `outputs/results/<dataset>/<método>/` | Predicciones (`predictions_{valid,test}.csv`, `*_results.json`), `runtime_*.json`, `manual_review_20.csv` | CSV/JSON livianos sí; ver excepciones |
| `data/iiyi/images_final/` | Imágenes (~1.2 GB) | **No** — copiar manual tras clonar |

### Excepciones que NO se versionan (`.gitignore`)
- Imágenes y cualquier `images_final.zip` (en cualquier ruta).
- Bajo `outputs/results/**`: `*.npy`, `*.npz`, `*.pt`, `*.bin`, `*.safetensors`
  y carpetas `final_adapter/` (matrices de similitud y checkpoints pesados).
- `checkpoints/`, `wandb/`, `runs/`, `config.yaml`, `.env`, caches.
- El antiguo `/results/` de la raíz (deprecado, anclado con `/` para no pisar
  `outputs/results/`).

## Resolución de imágenes

Las imágenes viven en `data/iiyi/images_final/images_{train,valid,test}/`. Todo
el código resuelve un `image_id` con `src.retrieval_utils.resolve_image_path()`,
que indexa esa carpeta recursivamente (con fallback al layout viejo
`data/images/`) y cachea el índice. Tanto los baselines de retrieval visual /
multimodal como el VLM resuelven imágenes por este camino, de forma consistente.

## Esquema de predicciones (común al equipo)

CSV con columnas: `split, encounter_id, image_id, question_es,
reference_answer_es, predicted_answer_es, model_name, dataset_variant, method`.

## Changelog estructural (2026-06-20)

- **README raíz creado** como punto de entrada del repo.
- **Convención unificada en `outputs/`.** Se migró
  `results/dataset_enriched/retrieval_textual/` →
  `outputs/metrics/dataset_enriched/retrieval_textual/` (con `git mv`, se
  preserva historial). El `results/` de la raíz quedó deprecado.
- **`.gitignore` corregido:** whitelisting explícito de `outputs/{datasets,metrics,results}`;
  se ignoran solo los artefactos pesados bajo `outputs/results/**`. Se ancló
  `/results/` para que no pisara `outputs/results/`. Se ignoró `images_final.zip`
  suelto en la raíz (~1 GB).
- **Path de imágenes arreglado:** `retrieval_utils.IMAGES_DIR` apuntaba a
  `data/images` (inexistente) → ahora `data/iiyi/images_final` con
  `resolve_image_path()` indexado. Verificado: 2944/2944 imágenes resueltas.
  Esto desbloquea retrieval visual y multimodal, que antes no encontraban
  ninguna imagen.

## Pendiente / propuesto (no ejecutado)

- **Fusionar scripts `*_retrieval.py` y `*_retrieval_short.py`** en uno solo con
  `--dataset longest|short` (como ya hace `evaluate_retrieval.py`). Reduce 10
  archivos casi duplicados a 5. En espera: implica borrar archivos y cambia el
  flujo de Damián, así que requiere acuerdo del equipo.
- **Archivos sueltos legacy** `outputs/dermavqa_{train,valid,test}_es_longest.{csv,jsonl}`
  (esquema viejo, una fila por imagen): están gitignored y se conservan en local
  por decisión explícita; el dataset canónico es `outputs/datasets/dataset_longest_answer.*`.
