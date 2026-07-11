# Snapshot comparativo actual

Este documento resume donde esta parada la comparacion experimental despues de
regenerar las tablas paper-ready.

CSV versionable asociado:

```text
outputs/paper/tables/paper_main_test_comparison.csv
```

La tabla se regenera con:

```bash
python -m src.evaluate_retrieval_heldout --dataset all
python -m src.build_paper_results
```

## Pregunta comparativa

> En DermaVQA-IIYI en espanol, conviene recuperar respuestas de casos similares
> o fine-tunear un VLM liviano con LoRA/QLoRA?

Comparaciones principales:

1. Retrieval textual sobre el dataset enriquecido.
2. VLM LoRA sobre el dataset enriquecido, agregado a nivel caso.
3. TF-IDF held-out train-only sobre respuesta larga y respuesta corta.
4. Qwen2.5-VL zero-shot vs Qwen2.5-VL LoRA sobre respuesta larga.

## Tabla principal actual

| Dataset | Metodo | Split | Unidad | n | sacreBLEU | chrF corpus | chrF mean | ROUGE-L | Token F1 | BERTScore F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Enriquecido | TF-IDF retrieval | test | caso | 100 | 8.266 | 26.990 | 0.286 | 0.216 | 0.256 | 0.716 |
| Enriquecido | E5 retrieval | test | caso | 100 | 8.555 | 30.091 | 0.308 | 0.212 | 0.259 | 0.714 |
| Enriquecido | SBERT retrieval | test | caso | 100 | 8.795 | 29.816 | 0.309 | 0.222 | 0.269 | 0.720 |
| Enriquecido | Qwen2.5-VL LoRA case-avg | test | caso | 100 | 5.022 | 34.532 | 0.368 | 0.250 | 0.314 | 0.737 |
| Respuesta larga | TF-IDF train-only | test | caso | 100 | 0.462 | 13.160 | 0.148 | 0.056 | 0.072 | - |
| Respuesta larga | Qwen2.5-VL zero-shot | test | caso | 100 | 0.409 | 20.548 | 0.182 | 0.072 | 0.092 | 0.628 |
| Respuesta larga | Qwen2.5-VL LoRA | test | caso | 100 | 0.328 | 11.082 | 0.157 | 0.112 | 0.131 | 0.667 |
| Respuesta corta | TF-IDF train-only | test | caso | 100 | 0.773 | 9.958 | 0.089 | 0.013 | 0.013 | - |

## Lectura actual

- En el dataset enriquecido, el VLM LoRA tiene mejor `chrF mean`, ROUGE-L,
  Token F1 y BERTScore que los retrieval textuales. Su `chrF corpus` tambien
  queda por encima, aunque su sacreBLEU corpus no supera a retrieval.
- En respuesta larga, LoRA mejora ROUGE-L, Token F1 y BERTScore frente a
  zero-shot, pero zero-shot mantiene mejor chrF.
- Los baselines TF-IDF train-only para respuesta larga/corta son leakage-free y
  sirven como piso reproducible. Los baselines SBERT/E5/visual/multimodal
  all-split quedan como contexto de apendice hasta regenerarlos held-out.

## Artefactos paper-ready

- `outputs/paper/tables/paper_main_test_comparison.csv`: tabla principal.
- `outputs/paper/tables/paper_all_metrics_long.csv`: tabla larga con valid/test,
  image-level y all-split legacy.
- `outputs/paper/tables/paper_missing_metrics_report.md`: huecos y caveats.
- `outputs/paper/tables/paper_clinical_review_20.csv`: hoja de revision clinica
  de 20 casos para completar manualmente.
- `outputs/paper/tables/paper_clinical_review_summary.csv`: resumen agregado de
  la revision preliminar AI.
- `outputs/paper/figures/*.svg`: figuras versionables para el paper.
- `survey/paper_results_interpretation.md`: lectura narrativa para resultados y
  discusion.

## Revision cualitativa preliminar

La hoja de 20 casos fue prellenada con etiquetas `ai_preliminary`. No reemplaza
la revision de un dermatologo, pero permite orientar el analisis de errores:

- correccion: 3 correctas, 9 parciales, 8 incorrectas;
- soporte diagnostico: 6 con soporte, 6 parciales, 8 sin soporte;
- recomendaciones: 5 sustentadas, 4 mayormente sustentadas, 9 no sustentadas y
  2 potencialmente inseguras;
- informacion inventada/alucinada: 1 sin evidencia, 5 menores, 6 parciales y
  8 claras;
- severidad preliminar: 5 baja, 7 media y 8 alta.

## Caveats restantes

- BERTScore no fue recomputado para los nuevos TF-IDF held-out, porque requiere
  dependencias/modelo mas pesados. Puede completarse luego si hace falta.
- Las metricas corpus del VLM enriquecido se calculan concatenando predicciones
  deduplicadas por caso cuando hay varias imagenes; no usa seleccion oracle.
- Falta confirmar manualmente la revision clinica prellenada: correccion,
  soporte diagnostico, seguridad, alucinacion, genericidad, contradicciones y
  tono en espanol.
