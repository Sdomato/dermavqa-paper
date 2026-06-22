# Snapshot comparativo actual

Este documento resume donde esta parada la comparacion experimental al cierre de
la corrida LoRA sobre `dataset_enriched`.

No es aun la tabla final del paper. Todavia falta normalizar algunas unidades de
evaluacion y completar revision manual.

CSV versionable asociado:

```text
outputs/metrics/final_model_comparison_snapshot.csv
```

## Comparacion principal planificada

La pregunta central del trabajo es:

> En DermaVQA-IIYI en espanol, conviene recuperar respuestas de casos similares
> o fine-tunear un VLM liviano con LoRA/QLoRA?

Comparaciones clave:

1. Retrieval textual/visual/multimodal sobre `dataset_longest_answer`.
2. VLM zero-shot vs VLM LoRA sobre `dataset_longest_answer`.
3. Retrieval textual vs VLM LoRA sobre `dataset_enriched`.
4. Comparacion final Santino vs Matias:
   - Santino: LoRA sobre `dataset_enriched`.
   - Matias: LoRA sobre `dataset_longest_answer`.

## Resultados disponibles

### Dataset enriquecido

| Metodo | Split | Unidad | n | chrF | ROUGE-L | Token F1 | sacreBLEU | BERTScore F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-IDF retrieval | test | caso | 100 | 26.990 corpus | - | - | 8.266 | 0.716 |
| E5-small retrieval | test | caso | 100 | 30.091 corpus | - | - | 8.555 | 0.714 |
| SBERT MiniLM retrieval | test | caso | 100 | 29.816 corpus | - | - | 8.795 | 0.720 |
| Qwen2.5-VL LoRA enriched | test | imagen | 314 | 0.365 mean / 35.691 corpus | 0.254 | 0.317 | 11.598 | 0.738 |

Lectura actual:

- El LoRA enriquecido supera a retrieval textual enriquecido en sacreBLEU,
  chrF corpus y BERTScore F1.
- La comparacion no es perfectamente justa todavia porque retrieval esta por
  caso y VLM por imagen.
- El resultado automatico es prometedor, pero hay ejemplos con recomendaciones
  o diagnosticos diferenciales posiblemente no sustentados por la referencia.

### Dataset respuesta larga

| Metodo | Split | Unidad | n | chrF mean | ROUGE-L | Token F1 | BERTScore F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Retrieval multimodal alpha=0.60 | all | caso | 998 | 0.165 | 0.084 | 0.082 | 0.657 |
| Retrieval textual SBERT | all | caso | 998 | 0.174 | 0.088 | 0.085 | 0.657 |
| Qwen2.5-VL zero-shot | test | caso | 100 | 0.182 | 0.072 | 0.092 | 0.628 |
| Qwen2.5-VL LoRA longest | test | caso | 100 | 0.157 | 0.112 | 0.131 | 0.667 |

Lectura actual:

- En `dataset_longest_answer`, zero-shot conserva mayor chrF que LoRA.
- LoRA mejora ROUGE-L, Token F1 y BERTScore F1 frente a zero-shot.
- Retrieval textual SBERT es competitivo en chrF y BERTScore, aunque no genera
  una respuesta nueva: devuelve la respuesta del vecino recuperado.

## Caveats de comparabilidad

- `dataset_enriched` se evalua por imagen; `dataset_longest_answer` se evalua
  por caso.
- Los targets son distintos: respuesta enriquecida sintetica vs respuesta
  original mas larga.
- Las metricas de retrieval enriquecido provienen del notebook textual y usan
  `chrF` corpus en escala 0-100; las metricas VLM reportan tambien `chrF mean`
  por fila en escala 0-1.
- `outputs/metrics/dataset_longest_answer/metrics_mixed.csv` no contiene aun
  las filas LoRA; para LoRA-longest se usaron los per-case CSV y
  `survey/vlm_experiments.md`.
- Falta una revision clinica manual comun para todos los modelos.

## Pendientes para tabla final

1. Regenerar una tabla unificada con:
   - dataset;
   - metodo;
   - target;
   - unidad de evaluacion;
   - split;
   - n;
   - chrF;
   - ROUGE-L;
   - Token F1;
   - BERTScore F1;
   - latencia;
   - costo aproximado.
2. Decidir si `dataset_enriched` se reporta por imagen, por caso o ambas.
3. Agregar revision manual de 20 ejemplos por modelo candidato final.
4. Hacer una comparacion cruzada opcional:
   - LoRA enriched evaluado contra respuesta larga;
   - LoRA longest evaluado contra respuesta enriquecida.
5. Consolidar la narrativa para el paper:
   - retrieval es barato, reproducible y competitivo;
   - LoRA enriched mejora metricas automaticas, pero exige auditoria clinica;
   - LoRA longest mejora seguimiento de estilo respecto de zero-shot.
