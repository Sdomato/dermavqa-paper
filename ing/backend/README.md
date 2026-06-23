# Backend — DermaAssist (Fase 1: retrieval)

API que, dada una consulta de paciente, devuelve los casos clínicos más
parecidos de la base con su respuesta como evidencia. No genera texto: solo
recupera casos existentes (cero riesgo de alucinación).

## Estructura

```
backend/
├── app/
│   ├── main.py          ← FastAPI: /health y /consulta
│   ├── config.py        ← settings por env var
│   ├── schemas.py       ← contrato de la API
│   └── retrieval/
│       ├── base.py      ← interfaz Retriever
│       ├── corpus.py    ← carga la base de casos (reusa src/retrieval_utils.py)
│       ├── tfidf.py     ← backend TF-IDF (default, liviano)
│       ├── e5.py        ← backend E5 (calidad paper, opcional)
│       └── factory.py   ← elige backend según config
└── tests/
```

## Correr en local

Desde la **raíz del repo** (el backend reutiliza `src/` y `outputs/`):

```bash
pip install -r ing/backend/requirements.txt
cd ing/backend
uvicorn app.main:app --reload
```

- API: http://localhost:8000
- Docs interactivas: http://localhost:8000/docs

### Probar

```bash
curl -s localhost:8000/health

curl -s localhost:8000/consulta \
  -H 'Content-Type: application/json' \
  -d '{"titulo":"Mancha roja en el brazo","contenido":"Pica desde hace una semana","k":3}'
```

### Frontend

Abrir `ing/frontend/index.html` en el navegador con el backend levantado.

## Con Docker

Desde `ing/`:

```bash
docker compose up --build
```

## Configuración (env vars)

| Variable | Default | Descripción |
| --- | --- | --- |
| `DERMA_RETRIEVER` | `tfidf` | Backend: `tfidf` (liviano) o `e5` (calidad paper) |
| `DERMA_TOP_K` | `5` | Casos similares por consulta |
| `DERMA_INDEX_SPLITS` | `all` | Splits que entran a la base (`all` o ej. `train`) |

> Para usar `e5` hay que instalar `torch` y `transformers` (ver `requirements.txt`).

## Tests

```bash
cd ing/backend
pytest
```
