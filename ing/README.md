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

🟡 **Fase 0 — Setup.** Definiendo scaffolding y alcance concreto. Ver
[`ROADMAP.md`](ROADMAP.md) para el plan por fases.

## Estructura prevista del código

```
ing/
├── README.md          ← este archivo
├── ROADMAP.md         ← plan por fases
├── backend/           ← API, servicios (retrieval, generación, seguridad)
├── frontend/          ← consola del dermatólogo
└── docs/              ← diagramas y notas de diseño
```

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

## Convenciones

- Rama principal de ingeniería: **`dev-ing`** (esta).
- Ramas de trabajo: `ing/<feature>` salidas de `dev-ing`.
- La investigación no se toca desde acá; si necesitamos un cambio en `src/`, se coordina
  con `develop`.
