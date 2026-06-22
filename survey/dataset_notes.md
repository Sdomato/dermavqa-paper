# Dataset notes

## Fuente

Usaremos el subconjunto IIYI de DermaVQA. El dataset contiene consultas de
pacientes, imagenes dermatologicas y respuestas medicas asociadas. Las consultas
estan disponibles en chino, ingles y espanol.

## Archivos observados

- `data/iiyi/train.json`: casos de entrenamiento.
- `data/iiyi/valid_ht.json`: casos de validacion.
- `data/iiyi/test_ht.json`: casos de test.
- `data/iiyi/test_ht_spanishtestsetcorrected.json`: variante corregida del test
  en espanol.
- `data/iiyi/images_final/`: imagenes dermatologicas.
- `data/iiyi/df_mediqa-m3g-final.csv`: metadatos por caso.
- `data/iiyi/df_userinfo.csv` y `data/iiyi/df_users_map.csv`: metadatos de
  autores y validacion.

## Conteos iniciales

- `train.json`: 842 casos.
- `valid_ht.json`: 56 casos.
- `test_ht.json`: 100 casos.
- Total observado: 998 casos.
- Imagenes encontradas en `images_final`: 2.945 archivos.
- Imagenes referenciadas por casos: 2.944.
- Imagenes extra no usadas: 1.

## Auditorias pendientes

- Identificar la imagen extra no referenciada por los casos.
- `split2encounterids.json` reporta 851 casos de train, pero `train.json`
  contiene 842. El puente entre IDs numericos y `ENC*` esta en
  `instanceid2encounterid.json`; hay 9 IDs de train en el split/mapa que no
  aparecen en los JSON/CSV finales.
- Revisar casos sin imagen, imagenes duplicadas y casos con multiples imagenes.
- Confirmar si `test_ht_spanishtestsetcorrected.json` debe reemplazar a
  `test_ht.json` para las evaluaciones en espanol.

## Construccion del target

No vamos a fijar una unica regla de target al principio. Como cada caso puede
tener multiples respuestas, prepararemos varias variantes de dataset y las
compararemos experimentalmente.

### Variante `longest_answer`

- pregunta = `query_title_es` + salto de linea + `query_content_es`;
- target = respuesta mas larga en `responses[*].content_es`;
- una fila por par imagen-pregunta-respuesta.

Ventaja: simple y barata. Riesgo: puede favorecer respuestas largas aunque no
sean las mas correctas.

### Variante `majority_answer`

- respuesta mayoritaria normalizada;
- usar solo cuando el acuerdo entre respuestas supere un umbral;
- dejar casos sin mayoria clara como `null` o moverlos a una variante separada.

Ventaja: aprovecha consenso. Riesgo: muchas respuestas son cortas o no
identicas aunque sean semanticamente compatibles.

### Variante `all_answers_raw`

- conservar todas las respuestas originales en una lista;
- preservar `author_id`, idioma y orden cuando esten disponibles;
- usar esta variante para analisis, prompts con ejemplos y evaluacion con
  multiples referencias.

Ventaja: maxima trazabilidad. Riesgo: no produce un unico target directo para
fine-tuning tradicional.

### Variante `llm_synthesized_answer`

- tomar todas las respuestas en espanol de un caso;
- pedirle a un LLM que las integre en un unico parrafo largo y coherente;
- usar creditos de Microsoft Azure para correr la sintesis;
- usar solo texto: pregunta del paciente y respuestas originales, nunca imagenes
  durante la sintesis;
- guardar tambien las respuestas originales para auditoria.
- deployment recomendado segun modelos disponibles en Azure for Students:
  `gpt-oss-120b`. Si Azure solo permite otro modelo de chat regional, usar ese
  deployment cambiando `AZURE_OPENAI_DEPLOYMENT`.

Script propuesto:

```bash
python src/build_llm_synthesized_dataset.py --dry-run --limit 10
python src/build_llm_synthesized_dataset.py --limit 30
python src/build_llm_synthesized_dataset.py
```

Variables de entorno:

```bash
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_DEPLOYMENT=gpt-oss-120b
AZURE_OPENAI_API_VERSION=2024-10-21
```

Para deployments de Azure AI Foundry v1, tambien se pueden usar:

```bash
AZURE_AI_FOUNDRY_ENDPOINT=https://...services.ai.azure.com/
AZURE_AI_FOUNDRY_KEY=...
AZURE_OPENAI_DEPLOYMENT=gpt-oss-120b
LLM_PROVIDER=foundry_v1
```

Prompt esperado:

```text
Recibiras varias respuestas medicas breves a una misma consulta dermatologica.
Unificalas en un solo parrafo en espanol. Usa solo informacion presente en las
respuestas originales. No agregues diagnosticos, tratamientos ni advertencias
nuevas. Si las respuestas se contradicen, menciona la incertidumbre de forma
breve.
```

Metadatos obligatorios de la sintesis:

- proveedor: Azure;
- deployment/modelo;
- version de prompt;
- temperatura y parametros de generacion;
- fecha de generacion;
- hash o identificador de las respuestas fuente;
- respuestas originales completas y respuestas deduplicadas usadas en el prompt;
- texto sintetizado;
- flags de contradiccion o baja confianza si se detectan.

Ventaja: genera targets mas ricos para entrenamiento y evaluacion. Riesgo: el
LLM puede introducir informacion no presente en las fuentes, por eso la sintesis
debe auditarse y configurarse con temperatura baja.

## Estadisticas a reportar

- casos por split;
- imagenes por caso;
- longitud de preguntas y respuestas;
- numero de respuestas por caso;
- acuerdo entre respuestas;
- distribucion de longitud por variante de target;
- tasa de casos con sintesis LLM valida;
- porcentaje de imagenes faltantes;
- distribucion de localizaciones anatomicas cuando el metadato este disponible.

## Cierre de `dataset_enriched`

Validado el 2026-06-16 como dataset enriquecido de Santino para baselines y
LoRA/QLoRA.

### Artefactos

- Dataset final: `outputs/datasets/dermavqa_iiyi_llm_synthesized_answer_finetune.jsonl`.
- CSV equivalente: `outputs/datasets/dermavqa_iiyi_llm_synthesized_answer_finetune.csv`.
- Paquete portable: `outputs/datasets/dermavqa_iiyi_llm_synthesized_answer_finetune.zip`.
- Contrato de columnas/conteos: `outputs/datasets/manifest.json`.
- Documentacion operativa: `outputs/datasets/README.md`.

### Generacion documentada

- Script revisado: `src/build_llm_synthesized_dataset.py`.
- Splits usados por el script: `train.json`, `valid_ht.json` y
  `test_ht_spanishtestsetcorrected.json`.
- Se genera una sintesis por caso (`encounter_id`) y luego se expande a una
  fila por imagen.
- El prompt usa solo `question_es` y las respuestas originales en espanol; no
  usa imagenes durante la sintesis.
- Las respuestas fuente se toman de `responses[*].content_es`, se filtran
  vacias y se deduplican antes de construir el prompt.
- Prompt version: `llm_synthesis_es_v5`.
- Deployment/modelo por defecto: `gpt-oss-120b`.
- Proveedor configurado por script: Azure OpenAI o Azure AI Foundry v1 segun
  endpoint/`LLM_PROVIDER`.
- Temperatura por defecto: `0.2`; max tokens por defecto: `900`.
- El script pide JSON estricto con `synthesized_answer_es`, `has_conflict`,
  `conflict_note` y `source_support_level`; el artefacto compacto final conserva
  solo las columnas necesarias para entrenamiento.

### Reglas principales del prompt

- Redactar una unica respuesta clinica concisa en espanol.
- Usar solo informacion presente en las respuestas originales.
- No agregar diagnosticos, tratamientos, advertencias, recomendaciones,
  morfologia, signos negativos ni hallazgos nuevos.
- Evitar respuestas de una sola palabra o solo etiqueta; convertir etiquetas en
  una frase natural cuando la fuente sea breve.
- Integrar incertidumbre si hay contradicciones y marcar conflicto solo ante
  contradiccion directa.
- Evitar frases meta sobre fuentes/autores y no mencionar que la respuesta fue
  sintetizada por un modelo.

### Conteos validados

| Split | Casos | Filas por imagen |
| --- | ---: | ---: |
| `train` | 842 | 2473 |
| `valid` | 56 | 157 |
| `test` | 100 | 314 |
| **Total** | **998** | **2944** |

Campos finales: `split`, `encounter_id`, `image_id`, `image_path`,
`question_es`, `answer_es`.

Validaciones realizadas:

- JSONL y CSV tienen 2944 filas y las mismas 6 columnas.
- Las parejas `(split, encounter_id, image_id)` coinciden exactamente con los
  JSON fuente.
- Preguntas vacias: 0.
- Respuestas vacias: 0.
- Rutas de imagen vacias: 0.
- Archivos de imagen faltantes en la maquina local: 0.
- Filas duplicadas completas: 0.
- Duplicados por `(split, encounter_id, image_id)`: 0.
- ZIP revisado con JSONL, CSV, README y manifest.

### Limitaciones y pendientes

- Las respuestas son sinteticas y requieren auditoria clinica manual antes de
  usarse como evidencia final.
- La sintesis es text-only; no valida ni corrige contra la imagen.
- La misma `answer_es` se repite para todas las imagenes del caso.
- `image_path` usa rutas absolutas locales, por lo que debe remapearse fuera de
  esta maquina.
- El artefacto compacto no conserva respuestas fuente ni flags de conflicto por
  fila; si se necesita auditoria fina, generar/exportar una version con
  metadatos completos.
- Costo aproximado pendiente: no se encontraron logs versionables de usage,
  tokens o costo de Azure.
