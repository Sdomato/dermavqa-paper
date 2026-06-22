# Interpretación de resultados para el paper

Este documento resume la lectura actual de los resultados paper-ready. Sirve
como base para redactar las secciones de resultados, discusión y limitaciones.

## Claim principal sugerido

El enriquecimiento textual de las respuestas originales, combinado con
fine-tuning LoRA/QLoRA de un VLM liviano, mejora la calidad automática de las
respuestas generadas frente a baselines de recuperación textual sobre el mismo
target enriquecido. Sin embargo, la mejora automática no elimina la necesidad
de auditoría clínica: en la revisión preliminar aparecen casos con diagnóstico
no respaldado, recomendaciones no sustentadas y cambios de entidad diagnóstica.

## Lectura cuantitativa

La tabla principal se encuentra en:

```text
outputs/paper/tables/paper_main_test_comparison.csv
```

En el dataset enriquecido, `Qwen2.5-VL-3B-Instruct + LoRA` agregado a nivel
caso obtiene:

- `chrF mean = 0.368`, por encima de los retrieval textuales enriquecidos
  (`0.286` a `0.309`);
- `ROUGE-L = 0.250` y `Token F1 = 0.314`, también por encima de TF-IDF/E5/SBERT;
- `BERTScore F1 = 0.737`, levemente superior a SBERT retrieval (`0.720`);
- `chrF corpus = 34.532`, por encima de los retrieval textuales enriquecidos.

La excepción es `sacreBLEU`: los retrieval textuales enriquecidos tienen mayor
sacreBLEU corpus que el VLM LoRA. Esto sugiere que el modelo fine-tuneado
parafrasea o reestructura más la respuesta, lo que puede beneficiar métricas
semánticas y chrF medio, pero no necesariamente BLEU.

En el dataset de respuesta larga, LoRA mejora frente a zero-shot en ROUGE-L,
Token F1 y BERTScore, pero no en chrF. Esto sugiere que el fine-tuning mejora
el ajuste semántico/estilístico al target, aunque no siempre aumenta el
solapamiento superficial.

## Lectura cualitativa preliminar

La hoja de revisión está en:

```text
outputs/paper/tables/paper_clinical_review_20.csv
```

La revisión ya está prellenada como `ai_preliminary` y debe ser confirmada por
un revisor humano/clinico antes de usarla como evidencia médica. Aun así, sirve
para orientar la discusión:

- En 20 casos, la revisión preliminar marcó 3 respuestas correctas, 9 parciales
  y 8 incorrectas.
- El soporte diagnóstico fue completo en 6 casos, parcial en 6 y ausente en 8.
- Las recomendaciones fueron sustentadas o mayormente sustentadas en 9 casos,
  no sustentadas en 9 y potencialmente inseguras en 2.
- La severidad preliminar del error fue baja en 5 casos, media en 7 y alta en 8.
- Hay casos claramente buenos, por ejemplo cuando el modelo conserva psoriasis,
  onicomicosis, foliculitis por Malassezia o urticaria papular.
- Hay errores clínicamente relevantes donde el modelo cambia la entidad central
  de la referencia: por ejemplo, linfangioma/nevus hacia psoriasis, angioma hacia
  granuloma anular, o nevus epidérmico pigmentado hacia verrugas planas.
- El modelo tiende a proponer confirmación diagnóstica o estudios como biopsia,
  prueba de hongos o pruebas de sangre. A veces esto está sustentado; otras
  veces aparece como recomendación no presente en la referencia.
- El español y el tono clínico suelen ser adecuados, pero la prudencia médica
  no siempre alcanza: algunas respuestas son plausibles en estilo pero no están
  bien sustentadas por el target.

## Cómo contarlo en resultados

Texto sugerido:

> En la comparación principal, el modelo multimodal fine-tuneado sobre respuestas
> enriquecidas obtuvo las mejores métricas medias de similitud léxica y semántica
> frente a los baselines textuales sobre el mismo target. La mejora fue más clara
> en chrF medio, ROUGE-L, token-F1 y BERTScore F1. Sin embargo, sacreBLEU corpus
> favoreció a retrieval, probablemente porque las respuestas recuperadas conservan
> más frases exactas del conjunto de referencia, mientras que el VLM genera
> reformulaciones.

> La revisión cualitativa preliminar mostró que el fine-tuning no elimina errores
> médicos: aunque el modelo adopta un estilo clínico en español y suele producir
> respuestas completas, en varios casos cambia el diagnóstico principal o agrega
> recomendaciones no sustentadas por la referencia. Por lo tanto, el sistema debe
> interpretarse como un experimento de adaptación multimodal y no como una
> herramienta clínica lista para uso médico.

## Limitaciones que conviene declarar

- Las respuestas enriquecidas fueron generadas por un LLM a partir de respuestas
  fuente; esto mejora consistencia, pero introduce una capa sintética en el target.
- Las métricas automáticas no miden seguridad clínica ni exactitud diagnóstica.
- La revisión cualitativa actual es preliminar y necesita confirmación humana.
- Algunas comparaciones usan targets distintos: respuesta corta, respuesta larga
  y respuesta enriquecida.
- En casos con varias imágenes, el dataset enriquecido expande un mismo target a
  múltiples imágenes. La tabla principal usa agregación por caso para reducir el
  sesgo por casos multi-imagen.
- Los baselines held-out limpios para respuesta larga/corta actualmente incluyen
  TF-IDF train-only; E5/SBERT/visual/multimodal held-out pueden agregarse como
  trabajo adicional si se desea una comparación más completa.

## Conclusión sugerida

El resultado más defendible es que el enriquecimiento del target y el fine-tuning
LoRA de un VLM pequeño son prometedores para mejorar respuestas dermatológicas en
español, especialmente frente a retrieval textual simple. No obstante, el modelo
aún requiere auditoría clínica y mecanismos de control antes de cualquier uso
asistencial.
