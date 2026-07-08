# Parte de Ingeniería — DermaAssist

![CI/CD](https://github.com/Sdomato/dermavqa-paper/actions/workflows/ci-ing.yml/badge.svg?branch=dev-ing)

Rama principal de la **parte de ingeniería** del proyecto. Acá vive el diseño y la
implementación del sistema; la **parte de investigación** (paper, experimentos, métricas)
sigue en la rama `develop`.

---

## Qué construimos

**DermaAssist**: un asistente que ayuda a un dermatólogo a responder consultas de pacientes
(texto en español + imágenes). Ante una consulta nueva, el sistema le arma al médico:

1. un **borrador de respuesta** fundamentado,
2. los **casos similares** que usó como evidencia,
3. **alertas de seguridad** sobre el borrador.

El dermatólogo edita y aprueba. **La respuesta nunca se envía sola al paciente.**

## Por qué este enfoque

El paper demostró que el modelo gana en métricas automáticas pero **no es clínicamente
confiable** (en la revisión: 15 % correcto, 40 % incorrecto, 10 % potencialmente inseguro).
Por eso no construimos un chatbot autónomo, sino un **copiloto del médico**: la IA acelera,
el humano garantiza la seguridad. Ese hallazgo del paper es la **restricción central de
diseño** del sistema.

---

## Cómo se relaciona con la investigación

La parte de ingeniería **reutiliza** los artefactos que ya están en `develop`:

| Artefacto en `develop` | Se usa como |
| --- | --- |
| `src/vlm_infer.py` (con `--adapter`) | Servicio de generación de borradores |
| Adapters LoRA (~160 MB) | Modelo que escribe los borradores |
| `src/multimodal_retrieval.py` + `retrieval_utils.py` | Servicio de retrieval / evidencia |
| `src/evaluate_predictions.py` | Métricas y monitoreo |
| Dataset IIYI (998 casos) | Base inicial de casos |

## Arquitectura (resumen)

```
Paciente → API (async) → ┬─ Retrieval (casos similares)
                         ├─ Generación RAG (VLM + casos como contexto)
                         └─ Capa de seguridad (verificador + flags)
                                   │
                                   ▼
                     Consola del dermatólogo (editar / aprobar)
                                   │
                                   ▼
                     Loop de mejora (feedback → índice + re-entrenamiento)
```

---

## Estado actual

🟢 **Fases 0, 1, 2, 3 y 4 — hechas.**
- **Fase 0** — scaffolding, FastAPI, Docker, CI/CD (tests + lint + imagen en GHCR).
- **Fase 1** — retrieval de casos similares: backends `tfidf` / `e5` / `multimodal`
  (E5+BiomedCLIP con cache de embeddings), endpoints de consulta (texto e imagen),
  frontend con fotos reales.
- **Fase 2** — borrador con RAG: cola asíncrona (`/borrador` → poll), generador
  `stub` (default) / `vlm` (Qwen2.5-VL+LoRA, opt-in).
- **Fase 3** — capa de seguridad: heurísticos + grounding + términos de riesgo + nivel,
  flujo de revisión médica (aprobar/editar/rechazar) y audit log. Validada contra los
  20 casos reales del paper (recall 100%).
- **Fase 4** — loop de mejora: cada borrador aprobado se vuelve caso recuperable en
  caliente (`app/feedback.py` + reindexado), dataset de validación clínica humana que
  crece con el uso (`GET /dataset/aprobados`), y tooling offline para el reentrenamiento
  periódico del LoRA (`scripts/build_finetune_dataset.py`, ver
  [`docs/fase4-reentrenamiento.md`](docs/fase4-reentrenamiento.md)).

### Calidad y efectividad de la respuesta

El objetivo del sistema es que el borrador que llega al médico sea **útil y seguro**
(no el texto más fluido: el paper mostró que fluido ≠ correcto). Sobre esa base:

- **Seguridad reforzada** (`app/safety/`) — además del grounding y los términos de
  riesgo, la capa señala los **dos modos de falla que documentó la revisión clínica**:
  *cambio de entidad diagnóstica* (el borrador desplaza el diagnóstico del caso más
  parecido) y *recomendaciones no sustentadas* (estudios/tratamientos ausentes en la
  evidencia). Ambos elevan el nivel de alerta para el revisor.
- **Harness de evaluación** (`GET /metricas`) — mide, desde el audit log, cuánto edita
  el médico el borrador (proxy de calidad), la tasa de aprobación y la distribución de
  niveles de seguridad. Da una señal, sin GPU, de si el sistema mejora con el uso.

🟢 **Roadmap completo.** Próximos incrementos posibles: capturar imágenes en la aprobación
para cerrar el loop también del VLM, y persistencia con trazabilidad real (hoy JSONL).
Ver [`ROADMAP.md`](ROADMAP.md) para el detalle por fase.

## Estructura del código

```
ing/
├── README.md          ← este archivo
├── ROADMAP.md         ← plan por fases (con estado)
├── docker-compose.yml ← levanta la API
├── docs/              ← handoffs y notas (ej. generación del cache de embeddings)
├── backend/           ← API FastAPI: retrieval + generación (ver backend/README.md)
│   ├── app/{retrieval,generation}/  · jobs.py (cola async)
│   └── scripts/       ← tooling offline (build_case_embeddings.py)
└── frontend/          ← consola del dermatólogo (index.html)
```

> Artefacto consumido por el servicio: `outputs/embeddings/case_embeddings.npz`
> (cache E5+BiomedCLIP de los 998 casos, generado offline).

---

## CI/CD

Pipeline en GitHub Actions (`.github/workflows/ci-ing.yml`):

- **CI** — en cada push y PR a `dev-ing` que toque `ing/` o `src/`, corre los tests del
  backend con `pytest`.
- **CD** — en push a `dev-ing` (si los tests pasan), construye la imagen Docker y la
  publica en **GHCR** como `ghcr.io/sdomato/dermavqa-assist-api:latest`.

Levantar la última imagen publicada:

```bash
docker run -p 8000:8000 ghcr.io/sdomato/dermavqa-assist-api:latest
```

## Modelo de ramas

La investigación y la ingeniería viven en ramas separadas, a propósito:

- **`develop`** — fuente de verdad de la **investigación**: paper, experimentos, métricas, y
  los **artefactos compartidos** que la ingeniería reutiliza (`src/`, el dataset en
  `outputs/datasets/`, los adapters LoRA).
- **`dev-ing`** — rama principal de **ingeniería** (DermaAssist). Salió de `develop` y
  construye `ing/` encima.

### Regla de oro: el flujo es en una sola dirección

```
develop  ───(merge)──▶  dev-ing
```

- Cuando `develop` actualiza algo que usamos (dataset, `src/`, adapters), se **mergea
  `develop` dentro de `dev-ing`** para traer lo nuevo:
  ```bash
  git checkout dev-ing && git merge origin/develop
  ```
- Lo de ingeniería (`ing/`, CI, Docker) **nunca** vuelve a `develop`.
- Si la ingeniería necesita un cambio en `src/`, se hace con un **PR chico a `develop`** y
  después se sincroniza — no se parchea suelto en `dev-ing`.

> Por qué: `dev-ing` depende de 3 artefactos de `develop` (`src/retrieval_utils.py`, el
> dataset, los adapters). Tratamos esos como una "interfaz publicada": solo cambian a
> propósito, y la sync deliberada evita que `dev-ing` se rompa en silencio.

### Convenciones

- Ramas de trabajo: `ing/<feature>` salidas de `dev-ing`, se mergean de vuelta a `dev-ing`.
- Todo lo que entra a `dev-ing` pasa por CI (tests + lint).
