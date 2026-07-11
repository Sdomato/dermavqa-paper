# Decisiones — correr DermaAssist con modelos reales

> Contexto: al preparar la demo de la parte de ingeniería quisimos pasar de la
> config liviana (`tfidf` + `stub`, pensada para test/CI sin GPU) a los **modelos
> reales**. Este documento registra qué se pudo, qué no, **por qué**, y cada
> decisión que tomamos, con sus trade-offs. La conclusión corta: el **retrieval real
> (E5) sí corre**; la **generación real (VLM + LoRA) no puede correr en esta máquina**
> por razones concretas, no por falta de ganas.

---

## 1. La pregunta central: ¿por qué no podemos usar el VLM + LoRA?

El generador "real" del paper es **Qwen2.5-VL-3B-Instruct** con un **adapter LoRA**
fine-tuneado sobre el dataset dermatológico. Localmente **no se puede ejecutar**, y hay
**tres razones independientes**, cada una de las cuales ya sería bloqueante por sí sola:

### Razón 1 — Los pesos del adapter LoRA no están en el repo (bloqueante principal)

El adapter fine-tuneado **no está versionado en ninguna rama**. Lo verificamos:

```bash
# En dev-ing y en develop:
find . -name "adapter_model.safetensors" -o -name "adapter_config.json"
# -> vacío. No hay pesos de adapter en ningún lado.

git ls-tree -r --name-only origin/develop | grep -iE "safetensors|adapter_config"
# -> vacío también.
```

Lo único que hay en `outputs/results/dataset_*/vlm_lora/` son los **resultados** de la
corrida de entrenamiento (predicciones `.csv`, métricas, logs, `runtime.json`), **no los
pesos**. El `.gitignore` excluye `outputs/*` salvo subdirectorios puntuales, y los
`.safetensors` (~160 MB) nunca se commitearon —lo cual es correcto: los pesos de modelo no
van a git.

**Consecuencia:** el adapter vive donde se entrenó (la **VM con GPU**). Sin esos pesos, el
`DERMA_GENERATOR=vlm` con LoRA no tiene qué cargar. No es un problema de configuración: es
que el artefacto no está presente.

### Razón 2 — La cuantización 4-bit (bitsandbytes) es solo CUDA

El código carga el VLM en 4-bit para que entre en memoria (`src/vlm_infer.py`,
`BitsAndBytesConfig(load_in_4bit=True)`). **bitsandbytes requiere GPU NVIDIA (CUDA)**: no
funciona en CPU ni en Apple Silicon (MPS). Esta máquina es una Mac:

```
cuda disponible: False
mps disponible:  True      # Metal, no sirve para bitsandbytes
```

Aunque tuviéramos el adapter, la ruta de carga por defecto fallaría en esta máquina.

### Razón 3 — Aun sorteando lo anterior, sería zero-shot, lento y frágil

Se podría intentar correr **Qwen base sin LoRA** (zero-shot), en fp16 sobre MPS, sin
cuantizar. Pero:

- Requiere instalar `peft` + `accelerate` + `qwen_vl_utils` (no están) y **descargar ~7 GB**
  de pesos base (con ~20 GB libres y HF sin token, throttleado).
- La inferencia de un VLM 3B en MPS es **lenta** (decenas de segundos por borrador).
- Y lo más importante: **no sería el modelo del paper**. Zero-shot ≠ LoRA fine-tuneado; el
  paper mostró que justamente el fine-tuning es lo que mueve las métricas. Correr el base
  daría una demo "de un VLM", no del sistema que describimos.

### Conclusión

El VLM + LoRA real corre **en la VM con GPU donde se entrenó**, no en esta Mac. La decisión
honesta para la demo local fue **dejar el generador en `stub`** y documentarlo, en vez de
fingir un modelo que no está o mostrar un zero-shot que no es el nuestro.

---

## 2. Qué SÍ pudimos correr: retrieval real con E5

El retrieval real **sí es viable localmente** y lo activamos:

```bash
DERMA_RETRIEVER=e5 make run
```

- **Por qué se puede y el VLM no:** E5 (`intfloat/multilingual-e5-base`, ~280 M params) es
  chico, corre en CPU sin cuantización, y sus pesos **se descargan de HuggingFace** (no
  dependen de un artefacto privado nuestro). Descarga ~1.1 GB una vez e indexa el corpus al
  arrancar.
- **Qué gana sobre TF-IDF:** matchea por **significado**, no por palabras. Verificado: la
  consulta *"placas rojas descamativas en codos y rodillas"* recupera como #1 un caso de
  **psoriasis** aunque la palabra "psoriasis" no aparezca en la consulta. Búsqueda ~50 ms.

Así, la demo corre con **modelo real en la mitad que se puede (retrieval)** y **stub honesto
en la mitad que no (generación)**.

---

## 3. Todas las decisiones tomadas (con trade-offs)

| # | Decisión | Por qué | Alternativa descartada |
|---|----------|---------|------------------------|
| 1 | **Retrieval = E5 real** | Modelo del paper, sin GPU, mejora semántica demostrable | Quedarse en TF-IDF (más pobre) |
| 2 | **Generación = stub, documentado** | El adapter LoRA no está local y bitsandbytes es CUDA-only | Fingir VLM o correr zero-shot (no es nuestro modelo) |
| 3 | **Frontend: mostrar el motivo real del error de revisión** | Antes ocultaba 404/409 tras un genérico; en demo confundía | Dejar el mensaje genérico |
| 4 | **Seguridad: match por prefijo + léxico ampliado** | El testeo halló falsos negativos ALTA (melanoma que no escalaba) | Enumerar cada conjugación a mano (frágil) |
| 5 | **Reindexado incremental (`add`) en vez de rebuild** | Con E5, aprobar re-embebía todo el corpus: ~160 s bloqueantes | Reindex en background (introduce race con búsquedas) |
| 6 | **`evidencia_debil`: documentar, no recalibrar a ciegas** | Con E5 el piso de similitud es ~0.9; el umbral 0.35 (tfidf) no dispara | Poner un número mágico sin datos de calibración |
| 7 | **Limpiar la data de test (con backup)** | Los agentes de testeo dejaron ~20 casos basura en el store | Pushear/demostrar con datos sucios |

### Detalle de las decisiones de código

**Decisión 4 — Recall clínica en banderas rojas.**
El análisis de seguridad matcheaba por palabra exacta (`\btermino\b`). El testeo mostró que
cuadros de alarma reales quedaban en nivel **bajo** por variantes lingüísticas: *"sangrando"*
(gerundio) no matcheaba `sangra`; *"asimétrico"* no matcheaba `asimetric` (regla muerta);
faltaban *"palma de la mano"*, *"no sana"*, *"llaga"*, *"evolucionó"*. Cambiamos el matcher de
banderas a **prefijo de palabra** (`\btermino`, captura conjugaciones) y ampliamos el léxico.
Criterio de diseño: en una capa de seguridad, **sobre-señalar es aceptable; dejar pasar un
melanoma no**. Sólo afecta a la detección de banderas; el grounding sigue con match exacto.

**Decisión 5 — Reindexado incremental.**
Al aprobar un borrador, el caso se suma a la base buscable. Antes se llamaba
`_rebuild_index()`, que **re-embebe todo el corpus**; con E5 eso son decenas de segundos y
**bloquea el request** (~160 s medidos con el corpus crecido, empeorando en cada aprobación).
Agregamos `Retriever.add(new_cases, all_cases)`: E5 lo overridea con un **append incremental**
(embebe sólo el caso nuevo, O(1)) y TF-IDF cae a un rebuild barato. Se serializa con las
búsquedas vía un `RLock` para que el índice y la lista de casos nunca se vean a medio
actualizar. Verificado: aprobar pasó de ~160 s a **0.06 s**, y el caso queda recuperable al
instante.

**Decisión 6 — Umbral de evidencia débil con E5.**
`evidencia_debil` se marca si la similitud del mejor caso `< DERMA_SIM_MIN` (default 0.35). Ese
umbral está calibrado para TF-IDF (donde una consulta sin match cae cerca de 0). Con E5 el
score es coseno mapeado a `[0,1]` mediante `(cos+1)/2`, cuyo **piso empírico es ~0.9** incluso
para texto irrelevante (probado con gibberish: sim ~0.91). Es decir, con E5 la regla
prácticamente no dispara. No lo "arreglamos" con un número mágico porque recalibrar bien
requiere datos; lo **documentamos** como limitación conocida y recomendamos, para producción
con E5, un umbral por-retriever o una señal relativa (margen del top-1 sobre la mediana).

---

## 4. Cómo se corren los modelos reales (referencia)

```bash
# Retrieval real (E5) — funciona local, sin GPU
DERMA_RETRIEVER=e5 make run

# Generación real (VLM + LoRA) — SOLO en VM con GPU CUDA y el adapter presente
DERMA_GENERATOR=vlm DERMA_ADAPTER_PATH=<ruta_al_final_adapter> make run
```

Ver también la sección "Correr con modelos reales" en
[`backend/README.md`](../backend/README.md).

---

## 5. Estado de la demo tras estas decisiones

- **Retrieval:** E5 real (calidad paper), verificado. 🟢
- **Generación:** stub (placeholder honesto anclado en el caso más parecido). El flujo
  completo —recuperar → borrador → seguridad → revisión → loop de mejora— funciona idéntico
  con stub o VLM; sólo cambia de dónde sale el texto del borrador. 🟢
- **Seguridad:** banderas rojas con mejor recall (fix del testeo). 🟢
- **Loop de mejora:** aprobar suma el caso al índice en caliente, instantáneo. 🟢
