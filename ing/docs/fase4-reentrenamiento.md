# Fase 4 — Loop de mejora: cómo se cierra el ciclo

> Objetivo de la fase: **que cada aprobación mejore el sistema.** Un borrador que
> un médico aprueba (o edita y aprueba) no se descarta: se vuelve conocimiento
> reutilizable.

El ciclo tiene dos velocidades, a propósito:

| Realimentación | Cuándo | Necesita GPU | Estado |
| --- | --- | --- | --- |
| **Retrieval** (el caso aprobado se vuelve evidencia recuperable) | en caliente, al aprobar | no | automático en el servicio |
| **LoRA** (reentrenar el modelo que escribe borradores) | periódico (batch) | sí | manual, en VM |

La separación es deliberada: lo barato y verificable corre en el servicio y está
cubierto por tests; lo caro (reentrenar el VLM) es un job offline que se dispara
cada tanto, cuando se juntaron suficientes casos.

---

## 1. Realimentación del retrieval (automática, sin GPU)

Cuando llega `POST /borrador/{job_id}/revision` con acción `aprobar` o `editar`:

1. Se registra la decisión en el audit log (Fase 3).
2. La consulta + la respuesta validada se guardan como **caso nuevo**
   (`app/feedback.py` → `casos_aprobados.jsonl`).
3. El índice se reconstruye en caliente (`_rebuild_index()` = casos base + aprobados),
   así que **una consulta parecida ya recupera ese caso** como evidencia.

Verificable a mano:

```bash
# Aprobar un borrador (ver flujo completo en backend/README.md), luego:
curl localhost:8000/health            # casos_aprobados sube, casos_indexados +1
curl localhost:8000/dataset/aprobados # el dataset humano-validado que crece con el uso
```

Cubierto por `tests/test_feedback.py` (incl. el *definition of done*: un caso
aprobado se recupera para su propia consulta).

---

## 2. Dataset de validación clínica humana

Cada caso aprobado es un par **consulta → respuesta validada por un médico**: justo
el dataset que al paper le falta (su revisión clínica fue de solo 20 casos). Se
expone en `GET /dataset/aprobados` y vive en `casos_aprobados.jsonl`.

A medida que el sistema se usa, este dataset crece **sin costo de anotación extra**:
la anotación es el trabajo normal del médico aprobando borradores.

---

## 3. Reentrenamiento del LoRA (periódico, en VM con GPU)

Cuando se acumularon suficientes casos aprobados (criterio operativo, ej. ≥ N
nuevos), se reentrena el adapter. **No corre en el servicio**: es el mismo
pipeline de la investigación (`src/train_longest.py`), en una VM con GPU.

### Paso 1 — exportar el dataset (offline, sin GPU)

```bash
cd ing/backend
python -m scripts.build_finetune_dataset \
    --aprobados .data/casos_aprobados.jsonl \
    --salida ../../outputs/datasets/aprobados_train.jsonl
```

Imprime cuántos casos hay y, de esos, cuántos tienen imagen (solo esos entran al
fine-tuning del VLM; ver nota abajo). El JSONL de salida usa las **mismas claves**
que el dataset del paper (`encounter_id`, `query_title_es`, `answer_es`,
`image_ids`, `_split="train_aprobado"`), para poder mergearlo al split de entrenamiento.

### Paso 2 — reentrenar en la VM (con GPU)

Misma receta que la corrida 2 documentada en `survey/vlm_experiments.md`
(Qwen2.5-VL-3B + LoRA, todas las imágenes por caso, early stopping, GPU L4). Se
suma `aprobados_train.jsonl` al split `train` y se corre `src/train_longest.py`.
El adapter resultante reemplaza al de `outputs/results/dataset_longest_answer/vlm_lora/final_adapter/`.

### Paso 3 — servir el adapter nuevo

El servicio levanta el VLM con el adapter vía variables de entorno
(`DERMA_GENERATOR=vlm`, `DERMA_ADAPTER_PATH=<ruta>`); apuntar al adapter
reentrenado y reiniciar. El retrieval ya venía mejorando solo, en cada aprobación.

---

## Nota: alcance actual (solo texto)

Hoy la promoción guarda la consulta y la respuesta aprobada (texto). El modelo de
datos **ya soporta `image_ids`**, pero el endpoint de revisión todavía no captura
las imágenes de la consulta original (se borran al terminar el job del borrador).
Mientras tanto:

- **Retrieval:** los casos aprobados mejoran el retrieval de texto sin condición.
- **LoRA:** `src/train_longest.py` descarta los items sin imagen, así que del
  retrain participan los casos aprobados **con** foto. Persistir las imágenes en la
  aprobación es el próximo incremento para cerrar el loop también del lado del VLM.
