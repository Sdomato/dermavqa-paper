# Roadmap — DermaAssist

Plan por fases de la parte de ingeniería. Cada fase es un **entregable cerrado** que se puede
demostrar por sí solo, y agrega una pieza de ingeniería sobre la anterior. El orden va de
**menor a mayor riesgo**: arrancamos por lo que no puede fallar peligrosamente.

---

## Fase 0 — Setup y scaffolding

> Objetivo: dejar el esqueleto del proyecto listo para que cualquiera lo levante.

- [ ] Estructura de carpetas (`backend/`, `frontend/`, `docs/`).
- [ ] Backend mínimo en FastAPI que levante y responda un *health check*.
- [ ] Definir el contrato de la API (qué entra: consulta + imágenes; qué sale: borrador + evidencia + flags).
- [ ] `docker-compose` para levantar todo con un comando (API + Postgres + Redis).
- [ ] Cargar el dataset IIYI como base inicial de casos.

**Definición de hecho:** `docker-compose up` levanta el backend y se puede pegarle al endpoint de salud.

---

## Fase 1 — Asistente de solo retrieval

> Objetivo: dada una consulta, devolver los casos más parecidos. **Cero riesgo de alucinación.**

- [ ] Servicio de retrieval que reusa `multimodal_retrieval.py` (texto + imagen, α=0.6).
- [ ] Indexar los 998 casos en FAISS o `pgvector`.
- [ ] Endpoint `POST /consulta` → devuelve los K casos similares con su respuesta.
- [ ] Frontend mínimo: el médico carga una consulta y ve la lista de casos similares.

**Definición de hecho:** se carga una consulta real y aparecen K casos relevantes con su texto e imagen.
**Por qué primero:** es útil desde el día uno y no puede dar una recomendación peligrosa (solo muestra casos existentes).

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
