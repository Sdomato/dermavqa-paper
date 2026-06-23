# Roadmap — DermaAssist

Plan por fases de la parte de ingeniería. Cada fase es un **entregable cerrado** que se puede
demostrar por sí solo, y agrega una pieza de ingeniería sobre la anterior. El orden va de
**menor a mayor riesgo**: arrancamos por lo que no puede fallar peligrosamente.

---

## Fase 0 — Setup y scaffolding ✅

> Objetivo: dejar el esqueleto del proyecto listo para que cualquiera lo levante.

- [x] Estructura de carpetas (`backend/`, `frontend/`).
- [x] Backend en FastAPI con *health check* (`GET /health`).
- [x] Contrato de la API definido (`app/schemas.py` + OpenAPI en `/docs`).
- [x] `docker-compose` para levantar la API con un comando. *(Postgres + Redis se suman en Fase 2/3, cuando hagan falta para la cola y el audit log.)*
- [x] Cargar el dataset IIYI (998 casos) como base inicial.
- [x] **Extra:** CI/CD en GitHub Actions (tests + lint + build/push de imagen a GHCR).

**Definición de hecho:** ✅ `docker compose up` levanta el backend y `/health` responde.

---

## Fase 1 — Asistente de solo retrieval ✅

> Objetivo: dada una consulta, devolver los casos más parecidos. **Cero riesgo de alucinación.**

- [x] Servicio de retrieval con interfaz intercambiable: **TF-IDF** (default), **E5** (texto, paper) y **multimodal** (E5+BiomedCLIP), reusando `src/`.
- [x] Backend **multimodal** (texto + imagen, α=0.6): consume un cache de embeddings precalculado (`outputs/embeddings/case_embeddings.npz`) y fusiona `α·texto + (1-α)·visual`. La matemática de fusión está testeada contra el cache real.
- [x] Indexar los 998 casos (índice **en memoria** por ahora; FAISS/`pgvector` cuando crezca la base).
- [x] Endpoints `POST /consulta` (texto) y `POST /consulta/imagen` (multipart, con foto) → K casos con respuesta, timing y exclusión de self.
- [x] Frontend: el médico carga una consulta (con foto opcional) y ve los casos similares **con sus fotos reales**.
- [x] **Extra:** endpoints `GET /casos/{id}` y `GET /imagen/{id}`, validación de input, **19 tests**, CI/CD.

**Definición de hecho:** ✅ se carga una consulta real y aparecen K casos relevantes con texto e imagen.
**Por qué primero:** es útil desde el día uno y no puede dar una recomendación peligrosa (solo muestra casos existentes).

> Nota: el backend `multimodal` necesita deps pesadas (`torch`/`open_clip`) + el cache `.npz` en
> el server. El default sigue siendo `tfidf` (liviano), así que el servicio arranca sin esas deps.

---

## Fase 2 — Borrador generado con RAG ✅

> Objetivo: agregar el borrador del VLM, **anclado** en los casos recuperados.

- [x] Generador intercambiable: `stub` (sin modelo, default/demo) y `vlm` (reusa `vlm_infer.py` con adapter LoRA, opt-in).
- [x] Prompt RAG: consulta + imágenes + los K casos recuperados como contexto, con instrucción anti-alucinación.
- [x] Manejo de la latencia (12–26 s): **cola asíncrona** (encolar → `job_id` → poll), con estado pending/running/done/error.
- [x] El borrador aparece en la consola junto a la evidencia (botón "Generar borrador" + poll en el frontend).
- [x] Tests del prompt, del stub y del flujo async completo (sin GPU).

**Definición de hecho:** ✅ el médico encola una consulta y recibe un borrador anclado en los casos, sin congelar la UI.

> Nota: el generador `vlm` necesita deps pesadas (`torch`/`transformers`/`peft`) + el adapter LoRA.
> El default es `stub`, así que el flujo completo es demoable y testeable sin GPU. *Streaming* del
> texto token-a-token queda como mejora futura (hoy el borrador llega completo al terminar el job).

---

## Fase 3 — Capa de seguridad ✅

> Objetivo: revisar el borrador antes de mostrarlo y marcar lo riesgoso.

- [x] Heurísticos automáticos: vacío / muy corto / **repetitivo** (degeneración del modelo).
- [x] Verificador (grounding): marca **diagnósticos del borrador ausentes en la evidencia** (el modo de falla del paper).
- [x] **Términos de riesgo**: detecta recomendaciones sensibles (biopsia, antibióticos, corticoides…).
- [x] **Nivel de riesgo** (bajo/medio/alto) en la respuesta de `/borrador`, mostrado en el frontend con badge + detalle.
- [x] **Flujo de aprobación**: botones editar / aprobar / rechazar (`POST /borrador/{id}/revision`) + **audit log** persistente (`GET /auditoria`).
- [x] Análisis 100% testeado (lógica pura) + **validación contra los 20 casos reales del paper (recall 100%)**.

**Definición de hecho:** ✅ un borrador que inventa un diagnóstico ausente en los casos aparece marcado, y toda decisión del médico queda registrada en el audit log (que es, además, el dataset de validación clínica humana que al paper le falta).

---

## Fase 4 — Loop de mejora

> Objetivo: que cada aprobación mejore el sistema.

- [ ] Guardar cada borrador aprobado/editado como caso nuevo en la base.
- [ ] Reindexar el retrieval con los casos aprobados.
- [ ] Acumular el dataset de **evaluación clínica humana real** (lo que al paper le falta).
- [ ] Re-entrenamiento periódico del adapter LoRA con los datos aprobados.

**Definición de hecho:** un caso aprobado hoy aparece como evidencia recuperable mañana, y existe un dataset humano-validado que crece con el uso.

---

## Riesgos y reglas de oro (transversales a todas las fases)

- **Nunca** una respuesta llega al paciente sin aprobación humana.
- Registro de auditoría inmutable (quién aprobó qué y cuándo).
- Privacidad de imágenes médicas.
- *Disclaimers* claros: el sistema asiste, no diagnostica.

---

## Orden sugerido para arrancar

Empezar por **Fase 0 + Fase 1**: con eso ya tenemos un sistema demostrable (retrieval de casos
similares) sobre el cual construir el resto sin riesgo clínico.
