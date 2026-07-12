# Backend — DermaAssist

API que asiste a un dermatólogo ante una consulta de paciente. Recupera los casos
clínicos más parecidos de la base (evidencia), arma un **borrador RAG** anclado en
ellos, lo pasa por una **capa de seguridad**, y registra la **revisión médica** que
retroalimenta el sistema. Cubre las Fases 1–4 del [ROADMAP](../ROADMAP.md); el
retrieval solo (Fase 1) no puede alucinar, y la generación queda siempre supeditada
a la aprobación humana.

## Estructura

```
backend/
├── app/
│   ├── main.py          ← FastAPI: endpoints (ver tabla abajo)
│   ├── config.py        ← settings por env var (con validación fail-fast)
│   ├── schemas.py       ← contratos de la API (request/response)
│   └── retrieval/
│       ├── base.py        ← interfaz Retriever
│       ├── corpus.py      ← carga la base de casos (reusa src/retrieval_utils.py)
│       ├── tfidf.py       ← backend TF-IDF (default, liviano)
│       ├── e5.py          ← backend E5 (texto, calidad paper)
│       ├── multimodal.py  ← backend multimodal E5+BiomedCLIP (consume el cache .npz)
│       └── factory.py     ← elige backend según config
├── scripts/
│   └── build_case_embeddings.py  ← genera el cache de embeddings (corre offline, GPU)
├── tests/               ← suite pytest (conftest + test_api)
├── Dockerfile
├── Makefile             ← atajos: install/run/test/lint/fmt
├── ruff.toml            ← config de lint
└── requirements.txt
```

## Endpoints

| Método | Ruta | Descripción |
| --- | --- | --- |
| `GET` | `/` | Redirige a `/docs` |
| `GET` | `/health` | Estado, versión, backend y nº de casos indexados |
| `POST` | `/consulta` | Solo-texto (JSON). Devuelve los K casos más similares (con respuesta y timing) |
| `POST` | `/consulta/imagen` | Multipart: texto + imágenes. El backend multimodal fusiona la señal visual |
| `POST` | `/borrador` | Encola un borrador RAG (recupera evidencia + genera). Devuelve `job_id` |
| `GET` | `/borrador/{job_id}` | Poll del borrador: `status`, `evidencia`, `borrador` y `seguridad` |
| `POST` | `/borrador/{job_id}/revision` | Registra la decisión médica (aprobar/editar/rechazar) en el audit log |
| `GET` | `/auditoria` | Lista las revisiones registradas (dataset de validación clínica humana) |
| `GET` | `/metricas` | Indicadores de calidad derivados del audit log (tasa de aprobación, edición media, niveles de seguridad) |
| `GET` | `/dataset/aprobados` | Dataset de casos aprobados por médicos que crece con el uso (Fase 4) |
| `GET` | `/casos/{encounter_id}` | Detalle de un caso de la base (404 si no existe) |
| `GET` | `/imagen/{image_id}` | Sirve la foto clínica de un caso (404 si no está en local) |

**Borrador (Fase 2):** `/borrador` recupera los casos similares, arma un prompt RAG anclado
en ellos (instrucción anti-alucinación) y genera un borrador para que un médico lo revise.
Es asíncrono porque el VLM tarda 12–26 s: encola y se hace poll. Por defecto usa el generador
`stub` (sin modelo, demoable); con `DERMA_GENERATOR=vlm` usa Qwen2.5-VL + LoRA.

**Seguridad y revisión (Fase 3):** cada borrador trae un análisis `seguridad` (nivel
bajo/medio/alto, diagnósticos no sustentados por la evidencia, términos de riesgo). El médico
decide con `/borrador/{id}/revision` (aprobar/editar/rechazar) y queda en un **audit log**
persistente (`/auditoria`), que acumula el dataset de validación clínica humana.

Contrato completo e interactivo: **http://localhost:8000/docs** (OpenAPI autogenerado).

## Correr en local

Desde la **raíz del repo** (el backend reutiliza `src/` y `outputs/`):

```bash
pip install -r ing/backend/requirements.txt
cd ing/backend
make run            # equivale a: uvicorn app.main:app --reload
```

- API: http://localhost:8000
- Docs interactivas: http://localhost:8000/docs

### Probar

```bash
curl -s localhost:8000/health

curl -s localhost:8000/consulta \
  -H 'Content-Type: application/json' \
  -d '{"titulo":"Mancha roja en el brazo","contenido":"Pica desde hace una semana","k":3}'

# Detalle de un caso e imagen
curl -s localhost:8000/casos/ENC00810
curl -s localhost:8000/imagen/IMG_ENC00810_00001.jpg --output foto.jpg
```

### Frontend

Abrir `ing/frontend/index.html` en el navegador con el backend levantado. Muestra los
casos similares con sus fotos reales (servidas por `/imagen/{id}`).

## Con Docker

Desde `ing/`:

```bash
docker compose up --build
```

O usar la imagen ya publicada por CI:

```bash
docker run -p 8000:8000 ghcr.io/sdomato/dermavqa-assist-api:latest
```

> Nota: la imagen no incluye las fotos clínicas (no versionadas), así que
> `/imagen/{id}` devolverá 404 dentro del contenedor salvo que se monten las imágenes.

## Configuración (env vars)

| Variable | Default | Descripción |
| --- | --- | --- |
| `DERMA_RETRIEVER` | `tfidf` | Backend: `tfidf` (liviano), `e5` (texto, paper) o `multimodal` (texto+imagen) |
| `DERMA_TOP_K` | `5` | Casos similares por consulta (default) |
| `DERMA_MAX_K` | `50` | Tope duro de `k` que un cliente puede pedir |
| `DERMA_INDEX_SPLITS` | `all` | Splits que entran a la base (`all` o ej. `train`) |
| `DERMA_CORS_ORIGINS` | `*` | Orígenes permitidos por CORS (coma-separados) |
| `DERMA_EMBEDDINGS_PATH` | `outputs/embeddings/case_embeddings.npz` | Cache de embeddings (backend multimodal) |
| `DERMA_ALPHA_TEXT` | `0.6` | Peso del texto en la fusión multimodal (`α·texto + (1-α)·visual`) |
| `DERMA_GENERATOR` | `stub` | Generador de borradores: `stub` (sin modelo) o `vlm` (Qwen2.5-VL + LoRA) |
| `DERMA_ADAPTER_PATH` | `` | Ruta del adapter LoRA (vacío = zero-shot). Solo `vlm` |
| `DERMA_VLM_MODEL` | `Qwen/Qwen2.5-VL-3B-Instruct` | Modelo base del generador `vlm` |
| `DERMA_MAX_NEW_TOKENS` | `256` | Tokens máximos del borrador |
| `DERMA_RAG_K` | `3` | Casos de evidencia que se pasan al generador como contexto |
| `DERMA_SIM_MIN` | `0.35` | Umbral de similitud: por debajo, la evidencia se marca como débil. Calibrado para `tfidf` (ver nota) |
| `DERMA_AUDIT_PATH` | `ing/backend/.data/revisiones.jsonl` | Archivo JSONL del audit log de revisiones |
| `DERMA_APROBADOS_PATH` | `ing/backend/.data/casos_aprobados.jsonl` | Archivo JSONL de casos aprobados que retroalimentan la base (Fase 4) |

> Para `e5` o `multimodal` hay que instalar `torch`/`transformers` (y `open_clip_torch` para
> multimodal). El backend `multimodal` además necesita el cache `.npz` (generarlo con
> `scripts/build_case_embeddings.py`, ver `ing/docs/handoff_embeddings_santino.md`).
> Config inválida (ej. retriever desconocido) hace que el servicio **no arranque** (fail-fast).

> **Nota sobre `DERMA_SIM_MIN` (evidencia débil):** el default `0.35` está calibrado para
> `tfidf`, donde una consulta sin casos parecidos cae cerca de 0. Con `e5` el score es coseno
> mapeado a `[0,1]` (`(cos+1)/2`), cuyo **piso empírico es ~0.9** incluso para texto irrelevante,
> así que el chequeo de `evidencia_debil` prácticamente no dispara con `e5`. Si se usa `e5`/
> `multimodal` en producción, hay que **recalibrar el umbral** para ese retriever (o pasar a una
> señal relativa, ej. margen del top-1 sobre la mediana). Limitación conocida, no un bug.

## Correr con modelos reales

Por defecto el servicio arranca liviano (`tfidf` + `stub`) para ser demoable y testeable sin
GPU. Para correr con los modelos reales:

### Retrieval real — E5 (búsqueda semántica del paper)

```bash
DERMA_RETRIEVER=e5 make run
```

La primera vez descarga `intfloat/multilingual-e5-base` (~1.1 GB) e indexa la base al arrancar
(unos minutos en CPU, una sola vez; luego queda en cache de HuggingFace). A diferencia de
TF-IDF, matchea por **significado**: una consulta de *"placas rojas descamativas en codos y
rodillas"* recupera casos de **psoriasis** aunque la palabra no aparezca en la consulta.
Verificado en `GET /health` → `"retriever": "e5"`, búsqueda ~50 ms.

### Retrieval multimodal — texto + imagen (E5 + BiomedCLIP)

Es el único modo en el que **la foto de la consulta influye en la búsqueda**. Con `tfidf`
y `e5` (solo texto) el frontend deja adjuntar una imagen y el endpoint la acepta, pero el
retriever la **ignora**. `multimodal` embebe la imagen con **BiomedCLIP** y la fusiona con
el texto: `score = 0.6·texto(E5) + 0.4·visual(BiomedCLIP)`.

```bash
pip install open_clip_torch pillow      # deps del encoder visual (una vez)
DERMA_RETRIEVER=multimodal make run
```

- **Corre en CPU, no requiere GPU** (a diferencia del VLM). Consume el cache de embeddings
  de los casos (`outputs/embeddings/case_embeddings.npz`, ya generado); no recalcula la base.
- La **primera** consulta carga E5 + descarga BiomedCLIP (~100 s una vez); después, ~60 ms.
- Verificado: consultando con la **foto propia** de un caso, ese caso vuelve como #1
  (sim 0.99); con el mismo texto pero sin foto, el top-1 es otro. La señal visual manda.

> **Limitación:** el índice visual sale del cache `.npz`, que solo tiene los 998 casos base.
> Los casos **aprobados por el loop de mejora** (Fase 4) no están en el cache, así que en modo
> `multimodal` **no son recuperables** (sí lo son con `tfidf`/`e5`, que embeben en vivo). Para
> sumarlos habría que regenerar el cache o embeber el caso aprobado al vuelo (mejora futura).

### Generación real — VLM (Qwen2.5-VL-3B + LoRA)

```bash
DERMA_GENERATOR=vlm DERMA_ADAPTER_PATH=<ruta_al_final_adapter> make run
```

> **Requiere GPU CUDA.** El generador `vlm` carga Qwen2.5-VL-3B en 4-bit (bitsandbytes), que
> **solo corre en GPU NVIDIA** — no en CPU ni en Apple Silicon (MPS). Además, el **adapter LoRA
> fine-tuneado no está versionado en este repo**: se entrenó en la VM con GPU y de esa corrida
> solo se trajeron los resultados (`outputs/results/dataset_*/vlm_lora/`: predicciones, métricas
> y logs), no los pesos (`adapter_model.safetensors`). Por eso la generación real corre **en la
> VM donde vive el adapter**; en local/demo el generador queda en `stub`, que arma el borrador a
> partir del caso más parecido (placeholder, sin modelo). El flujo completo —recuperar →
> borrador → análisis de seguridad → revisión médica → loop de mejora— es idéntico con `stub` o
> `vlm`; lo único que cambia es de dónde sale el texto del borrador.

## Tests y lint

```bash
cd ing/backend
make test     # pytest
make lint     # ruff check
make fmt      # ruff format
```

Ambos (`ruff` + `pytest`) corren automáticamente en CI en cada push/PR a `dev-ing`.
