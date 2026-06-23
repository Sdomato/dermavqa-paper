# Backend — DermaAssist (Fase 1: retrieval)

API que, dada una consulta de paciente, devuelve los casos clínicos más
parecidos de la base con su respuesta como evidencia. No genera texto: solo
recupera casos existentes (cero riesgo de alucinación).

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
| `GET` | `/casos/{encounter_id}` | Detalle de un caso de la base (404 si no existe) |
| `GET` | `/imagen/{image_id}` | Sirve la foto clínica de un caso (404 si no está en local) |

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

> Para `e5` o `multimodal` hay que instalar `torch`/`transformers` (y `open_clip_torch` para
> multimodal). El backend `multimodal` además necesita el cache `.npz` (generarlo con
> `scripts/build_case_embeddings.py`, ver `ing/docs/handoff_embeddings_santino.md`).
> Config inválida (ej. retriever desconocido) hace que el servicio **no arranque** (fail-fast).

## Tests y lint

```bash
cd ing/backend
make test     # pytest
make lint     # ruff check
make fmt      # ruff format
```

Ambos (`ruff` + `pytest`) corren automáticamente en CI en cada push/PR a `dev-ing`.
