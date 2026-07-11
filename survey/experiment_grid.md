# Experiment grid

## Pregunta central

Comparar adaptacion parametricamente eficiente con LoRA/QLoRA contra
recuperacion basada en casos para VQA dermatologico en espanol.

## Condiciones

| ID | Condicion | Entrada | Salida |
| --- | --- | --- | --- |
| T0 | TF-IDF textual | Pregunta ES | Respuesta del caso recuperado |
| T1 | Embeddings textuales | Pregunta ES | Respuesta del caso recuperado |
| V0 | Retrieval visual | Imagen | Respuesta del caso recuperado |
| M0 | Retrieval multimodal | Pregunta ES + imagen | Respuesta del caso recuperado |
| Z0 | VLM zero-shot | Pregunta ES + imagen | Respuesta generada |
| L0 | VLM LoRA/QLoRA | Pregunta ES + imagen | Respuesta generada |
| H0 | Hibrido opcional | Pregunta ES + imagen + casos recuperados | Respuesta generada |

## Variantes de target

Cada condicion debe poder evaluarse contra mas de una variante del dataset:

| Variante | Target | Uso principal |
| --- | --- | --- |
| `longest_answer` | Respuesta mas extensa | Baseline de preparacion simple |
| `majority_answer` | Respuesta con mayor acuerdo | Evaluar consenso entre expertos |
| `all_answers_raw` | Lista completa de respuestas | Multiple-reference evaluation y auditoria |
| `llm_synthesized_answer` | Parrafo consolidado por LLM en Azure | Fine-tuning/evaluacion con target mas rico |

La variante sintetizada por LLM debe generarse antes de los experimentos finales
y versionarse como un artefacto del pipeline, no editarse manualmente.

## Parametros a barrer

- `k`: 1, 3, 5, 10 para recuperacion.
- `alpha`: 0.0, 0.25, 0.5, 0.75, 1.0 para fusion multimodal.
- prompt VLM: instruccion corta vs instruccion con cautela medica.
- LoRA rank: valores chicos primero, por ejemplo 8 o 16.
- entrenamiento: pocas epocas, batch pequeno y gradient accumulation.
- target del entrenamiento: `longest_answer` vs `llm_synthesized_answer`.

## Orden sugerido

1. Preparar dataset procesado y splits finales.
2. Generar variantes `longest_answer`, `majority_answer` y `all_answers_raw`.
3. Generar `llm_synthesized_answer` usando Azure.
4. Correr TF-IDF para tener baseline rapido.
5. Agregar embeddings textuales multilingues.
6. Agregar retrieval visual.
7. Fusionar texto e imagen con barrido de `alpha`.
8. Correr zero-shot con el VLM elegido.
9. Fine-tunear con LoRA/QLoRA si hay GPU suficiente.
10. Evaluar y consolidar resultados por variante de target.

## Estado actual de ejecucion

| Bloque | Estado | Artefactos |
| --- | --- | --- |
| Datasets `longest_answer` y `short_answer` | hecho | `outputs/datasets/dataset_longest_answer.*`, `outputs/datasets/dataset_short_answer.*` |
| Dataset `llm_synthesized_answer` / `dataset_enriched` | hecho | `outputs/datasets/dermavqa_iiyi_llm_synthesized_answer_finetune.zip` |
| Retrieval textual enriched | hecho | `outputs/metrics/dataset_enriched/retrieval_textual/` |
| Retrieval long/short textual, visual y multimodal | hecho | `outputs/metrics/dataset_longest_answer/`, `outputs/metrics/dataset_short_answer/` |
| VLM zero-shot longest | hecho | `outputs/results/dataset_longest_answer/vlm_zero_shot/` |
| VLM LoRA longest | hecho | `outputs/results/dataset_longest_answer/vlm_lora/` |
| VLM LoRA enriched | hecho | `outputs/results/dataset_enriched/vlm_lora/` |
| Revision clinica manual | pendiente | plantilla/criterios a consolidar |
| Comparacion final paper | pendiente | `survey/final_comparison_snapshot.md` como base |

## Tabla final esperada

| Metodo | ROUGE-L | F1 token | BERTScore | Recall@1 | Recall@5 | MRR | Notas |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TF-IDF | TBD | TBD | TBD | TBD | TBD | TBD | Baseline inicial |
| Text embeddings | TBD | TBD | TBD | TBD | TBD | TBD | Semantico |
| Visual retrieval | TBD | TBD | TBD | TBD | TBD | TBD | Solo imagen |
| Multimodal retrieval | TBD | TBD | TBD | TBD | TBD | TBD | Mejor `alpha` |
| VLM zero-shot | TBD | TBD | TBD | N/A | N/A | N/A | Sin fine-tuning |
| VLM LoRA/QLoRA | TBD | TBD | TBD | N/A | N/A | N/A | Adaptado |
| Hibrido | TBD | TBD | TBD | TBD | TBD | TBD | Opcional |
