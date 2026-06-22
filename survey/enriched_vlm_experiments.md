# Experimentos VLM: LoRA Fine-tuning sobre `dataset_enriched`

Este documento registra la corrida de Santino para entrenar un VLM sobre el
dataset enriquecido con respuestas sintetizadas por LLM.

## Resumen ejecutivo

- **Dataset**: `dataset_enriched`.
- **Modelo base**: `Qwen/Qwen2.5-VL-3B-Instruct`.
- **Metodo**: QLoRA 4-bit + LoRA.
- **Entrada**: imagen + `question_es`.
- **Target**: `answer_es` enriquecida.
- **Unidad de entrenamiento**: una fila por imagen; si un caso tiene varias
  imagenes, se repite la misma respuesta enriquecida para cada imagen.
- **Entrenamiento real**: completado en Google Cloud con GPU NVIDIA L4.
- **Resultados versionados**:
  - `outputs/results/dataset_enriched/vlm_lora/`
  - `outputs/metrics/dataset_enriched/metrics_mixed.csv`
- **Pesos no versionados**:
  - `outputs/results/dataset_enriched/vlm_lora/final_adapter/`
  - `outputs/results/dataset_enriched/vlm_lora/checkpoints/`

## Dataset usado

Artefacto portable:

```text
outputs/datasets/dermavqa_iiyi_llm_synthesized_answer_finetune.zip
```

Conteos:

| Split | Casos | Filas por imagen |
| --- | ---: | ---: |
| train | 842 | 2473 |
| valid | 56 | 157 |
| test | 100 | 314 |
| **Total** | **998** | **2944** |

Columnas finales:

```text
split, encounter_id, image_id, image_path, question_es, answer_es
```

## Configuracion de entrenamiento

| Parametro | Valor |
| --- | --- |
| Modelo base | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Metodo | QLoRA 4-bit |
| Epochs corridas | 1 |
| Batch size | 1 |
| Gradient accumulation | 16 |
| Learning rate | 2e-4 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Scheduler | cosine |
| Precision | bfloat16 si esta disponible |
| Max image pixels | `256 * 28 * 28` |
| Max new tokens inferencia | 256 |

Script principal:

```bash
bash scripts/run_enriched_vlm_lora.sh --epochs 1
```

El script ejecuta:

1. `python3 -m src.train_enriched`
2. `python3 -m src.vlm_infer_enriched --split valid --adapter ...`
3. `python3 -m src.vlm_infer_enriched --split test --adapter ...`
4. `python3 -m src.evaluate_predictions ...`

## Resultados operativos

| Metrica operativa | Valor |
| --- | ---: |
| Train examples | 2473 |
| Valid examples | 157 |
| Global steps | 155 |
| Tiempo de entrenamiento | 4636.4 s (77.3 min) |
| VRAM pico | 6.73 GB |
| Tamano adapter | 160.1 MB |
| Best checkpoint | `checkpoint-100` |
| Eval loss valid | 1.8444 |
| Eval token accuracy valid | 0.5911 |

Inferencia con adapter:

| Split | n | Latencia media | Tiempo total |
| --- | ---: | ---: | ---: |
| valid | 157 | 15.96 s/ejemplo | 41.8 min |
| test | 314 | 14.97 s/ejemplo | 78.3 min |

## Metricas de generacion

Fuente: `outputs/metrics/dataset_enriched/metrics_mixed.csv`.

| Split | n | Empty | chrF mean | ROUGE-L | Token F1 | sacreBLEU | chrF corpus | BERTScore F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| valid | 157 | 0 | 0.361 | 0.243 | 0.305 | 10.904 | 35.705 | 0.737 |
| test | 314 | 0 | 0.365 | 0.254 | 0.317 | 11.598 | 35.691 | 0.738 |

## Comparacion contra retrieval textual enriquecido

El retrieval textual enriquecido se evaluo por caso (`valid=56`, `test=100`),
mientras que el VLM enriquecido se evaluo por imagen (`valid=157`, `test=314`).
Por eso la comparacion es orientativa hasta normalizar por `encounter_id`.

| Metodo | Split | n | sacreBLEU | chrF corpus | BERTScore F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| TF-IDF retrieval | test | 100 | 8.266 | 26.990 | 0.716 |
| E5-small retrieval | test | 100 | 8.555 | 30.091 | 0.714 |
| SBERT MiniLM retrieval | test | 100 | 8.795 | 29.816 | 0.720 |
| Qwen2.5-VL LoRA enriched | test | 314 | 11.598 | 35.691 | 0.738 |

Lectura preliminar: el LoRA enriquecido supera a los baselines textuales
enriquecidos en las metricas automaticas principales disponibles, pero todavia
necesita auditoria manual porque puede generar recomendaciones no sustentadas.

## Observaciones cualitativas iniciales

Ejemplos inspeccionados muestran dos comportamientos:

- El modelo suele responder con formato clinico conciso y usa frases prudentes
  como "compatible con" o "tambien se consideran diagnosticos diferenciales".
- En algunos casos agrega recomendaciones de tratamiento o biopsia que no
  necesariamente estaban en la referencia enriquecida. Esto debe marcarse como
  riesgo de alucinacion o exceso clinico en la revision manual.

Ejemplos que conviene revisar manualmente:

| Split | Encounter | Observacion |
| --- | --- | --- |
| valid | `ENC00853` | Predice eczema/psoriasis y agrega tratamiento con corticoides/antihistaminicos. |
| valid | `ENC00854` | La referencia apunta a tinea manus/eccema herpetico; la prediccion deriva a eczema/psoriasis y manejo general. |
| test | `ENC00908` | La referencia incluye foliculitis/sifilis/dishidrosis; la prediccion responde urticaria papular y sugiere patologia. |
| test | `ENC00909` | La referencia menciona linfangioma/nevus/penfigoide; la prediccion menciona psoriasis/verrugas y biopsia. |
| test | `ENC00910` | La referencia sugiere confirmar hongos; la prediccion se acerca parcialmente al pedir prueba de hongos. |

## Artefactos

| Artefacto | Estado | Notas |
| --- | --- | --- |
| `predictions_valid.csv` | versionado | 157 filas |
| `predictions_test.csv` | versionado | 314 filas |
| `metrics_mixed.csv` | versionado | resumen valid/test |
| `per_case_vlm_lora_valid.csv` | versionado | metricas por fila |
| `per_case_vlm_lora_test.csv` | versionado | metricas por fila |
| `train_runtime.json` | versionado | metricas operativas |
| `training_log_history.csv/json` | versionado | curva de entrenamiento |
| `final_adapter/` | local, no GitHub | ~153 MB descargado |
| `checkpoints/` | local, no GitHub | ~897 MB descargado |

## Pendientes

1. Hacer revision clinica manual de 20 casos con etiquetas:
   - correcto/parcial/incorrecto;
   - informacion inventada;
   - recomendacion no sustentada;
   - respuesta demasiado generica;
   - espanol/tono clinico.
2. Normalizar comparacion por `encounter_id` para comparar retrieval y VLM con
   el mismo numero de unidades.
3. Decidir si se reporta evaluacion por imagen, por caso, o ambas.
4. Agregar costo real si Google Billing exporta el gasto final de la VM.
5. Integrar estos resultados en la tabla final del paper.
