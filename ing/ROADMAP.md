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

## Fase 2 — Borrador generado con RAG

> Objetivo: agregar el borrador del VLM, **anclado** en los casos recuperados.

- [ ] Servicio de generación que reusa `vlm_infer.py` con el adapter LoRA.
- [ ] Armar el prompt RAG: consulta + imágenes + los K casos recuperados como contexto.
- [ ] Manejo de la latencia (12–26 s): cola asíncrona + estado "generando…" + *streaming* del texto.
- [ ] El borrador aparece en la consola junto a la evidencia.

**Definición de hecho:** el médico ve un borrador coherente que cita/usa los casos recuperados, sin congelar la UI.

---

## Fase 3 — Capa de seguridad

> Objetivo: revisar el borrador antes de mostrarlo y marcar lo riesgoso.

- [ ] Heurísticos automáticos: vacío / demasiado corto / demasiado genérico (reusar flags `auto_*`).
- [ ] Verificador: comparar el borrador contra los casos recuperados y marcar **afirmaciones no sustentadas**.
- [ ] Resaltar en la UI los diagnósticos/tratamientos riesgosos (biopsia, antibióticos, etc.).
- [ ] Flujo de aprobación: botones **editar / aprobar / rechazar** + registro de auditoría.

**Definición de hecho:** un borrador que inventa un diagnóstico ausente en los casos aparece marcado en rojo, y toda acción del médico queda registrada.

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
