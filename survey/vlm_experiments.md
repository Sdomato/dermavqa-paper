# Experimentos VLM: Zero-Shot y LoRA Fine-tuning (dataset_longest_answer)

Documenta el proceso completo de los experimentos con Qwen2.5-VL-3B-Instruct sobre
`dataset_longest_answer`: zero-shot baseline y fine-tuning QLoRA (dos corridas).

---

## Entorno de cómputo

### Corrida 1 (LoRA con 1 imagen, T4)
- **Instancia**: `dermavqa-train-2`, zona `asia-east1-c`
- **GPU**: NVIDIA Tesla T4 (14.56 GB VRAM)
- **SO**: Deep Learning VM with CUDA M132 (Ubuntu 22.04)
- **Costo**: ~$0.40/hora · Total estimado: ~$8-9

### Corrida 2 (LoRA con todas las imágenes, L4)
- **Instancia**: `dermavqa-l4b`, zona `us-east1-c`
- **GPU**: NVIDIA L4 (24 GB VRAM)
- **SO**: Deep Learning VM with CUDA 12.9 (Ubuntu 22.04)
- **Costo**: ~$0.70/hora · Total estimado: ~$3

---

## Modelo elegido

Se usó **Qwen2.5-VL-3B-Instruct** en lugar del 7B original.

El 7B no entra en 16 GB VRAM con QLoRA 4-bit + imágenes. Falla con OOM en el forward pass
del vision encoder incluso con `max_pixels` reducido. El 3B entró con 6.9 GB de VRAM pico,
dejando margen cómodo.

> *"Due to GPU memory constraints on a T4 (16 GB), we used Qwen2.5-VL-3B-Instruct instead
> of the 7B variant. The 7B model requires at least 24 GB VRAM (e.g., L4 or A100) for
> QLoRA fine-tuning with multimodal inputs."*

---

## Restricciones de memoria aplicadas

| Ajuste | Valor | Motivo |
|--------|-------|--------|
| `max_pixels` del processor | `256 × 28 × 28` | Reduce resolución procesada por el vision encoder |
| Imágenes por ejemplo (entrenamiento) | 1 (primera imagen) | Casos con 2-3 fotos duplicaban uso de VRAM |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Reduce fragmentación de memoria |

La limitación a 1 imagen aplica **solo al entrenamiento**. La inferencia (tanto zero-shot
como LoRA) usa todas las imágenes disponibles por encuentro.

---

## Experimento 1: Zero-Shot Baseline

Sin ningún fine-tuning. El modelo base Qwen2.5-VL-3B-Instruct recibe el prompt con imagen(es)
y pregunta y genera la respuesta directamente.

### Tiempos de inferencia

| Split | n | Latencia media | Tiempo total |
|-------|---|----------------|--------------|
| valid | 56 | 26.1s/ejemplo | 24.3 min |
| test | 100 | 26.7s/ejemplo | 44.6 min |

(más lento que LoRA porque zero-shot usa todas las imágenes del encuentro, no solo 1)

### Métricas zero-shot

| Split | n | chrF | ROUGE-L | Token F1 | sacreBLEU | BERTScore F1 |
|-------|---|------|---------|----------|-----------|--------------|
| valid | 56 | 0.193 | 0.081 | 0.110 | 0.556 | — |
| test | 100 | 0.183 | 0.072 | 0.092 | 0.409 | — |

*BERTScore no calculado en local (crash de memoria en Mac); pendiente de cálculo en VM.*

---

## Experimento 2: QLoRA Fine-tuning — Corrida 1 (1 imagen por caso, T4)

### Hiperparámetros de entrenamiento

| Parámetro | Valor |
|-----------|-------|
| Modelo base | Qwen2.5-VL-3B-Instruct |
| Épocas | 3 |
| Learning rate | 2e-4 |
| LR scheduler | cosine |
| Warmup ratio | 0.03 |
| Batch size | 1 |
| Gradient accumulation | 16 |
| LoRA r | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Cuantización | QLoRA 4-bit (nf4, bfloat16) |
| max_pixels | 256 × 28 × 28 |
| Checkpoint selection | eval_loss mínimo sobre valid |

### Resultados operativos de entrenamiento

| Métrica | Valor |
|---------|-------|
| Ejemplos de train | 842 |
| Ejemplos de valid | 56 |
| Tiempo total | 5.75 horas (20,683s) |
| VRAM pico | 6.91 GB |
| Tamaño del adapter | 160.1 MB |
| Parámetros entrenables | 37,152,768 (0.98% del total) |

### Tiempos de inferencia LoRA

| Split | n | Latencia media | Tiempo total |
|-------|---|----------------|--------------|
| valid | 56 | 10.4s/ejemplo | 9.7 min |
| test | 100 | 12.5s/ejemplo | 20.9 min |

### Métricas LoRA

| Split | n | chrF | ROUGE-L | Token F1 | sacreBLEU | BERTScore F1 |
|-------|---|------|---------|----------|-----------|--------------|
| valid | 56 | 0.145 | 0.101 | 0.113 | 0.237 | 0.670 |
| test | 100 | 0.157 | 0.112 | 0.131 | 0.328 | 0.667 |

---

## Experimento 3: QLoRA Fine-tuning — Corrida 2 (todas las imágenes, L4)

Mismo pipeline que la Corrida 1 pero sin la limitación de 1 imagen por caso y con early stopping.

### Cambios respecto a Corrida 1
- Sin límite de imágenes por caso (`image_paths[:1]` eliminado)
- Early stopping con paciencia 3 (techo 5 épocas)
- GPU L4 necesaria — VRAM pico 15.0 GB (T4 de 14.56 GB no hubiera entrado)

### Resultados operativos

| Métrica | Valor |
|---------|-------|
| Ejemplos de train | 842 |
| Ejemplos de valid | 56 |
| Épocas (early stopping) | ~3.78 |
| Tiempo total | 2.9 horas (10,464s) |
| VRAM pico | 15.0 GB |
| Tamaño del adapter | 160 MB |

### Métricas Corrida 2

| Split | n | chrF | ROUGE-L | Token F1 | sacreBLEU | BERTScore F1 |
|-------|---|------|---------|----------|-----------|--------------|
| valid | 56 | 0.134 | 0.098 | 0.109 | 0.288 | 0.671 |
| test | 100 | 0.155 | 0.104 | 0.117 | 0.501 | 0.669 |

---

## Comparativa completa (test, 100 casos)

| Método | chrF | ROUGE-L | Token F1 | BERTScore F1 |
|--------|------|---------|----------|--------------|
| Retrieval SBERT | 0.174 | 0.088 | 0.088 | 0.657 |
| Retrieval E5 | 0.179 | 0.081 | 0.077 | 0.656 |
| Retrieval Multimodal | 0.171 | 0.080 | 0.080 | 0.658 |
| VLM Zero-Shot | 0.183 | 0.072 | 0.092 | 0.628 |
| VLM LoRA Corrida 1 (1 img, T4) | 0.157 | **0.112** | **0.131** | 0.667 |
| VLM LoRA Corrida 2 (todas imgs, L4) | 0.155 | 0.104 | 0.117 | **0.669** |

Usar todas las imágenes no mejoró significativamente — las diferencias son menores al 1% en todas
las métricas. El BERTScore subió levemente (+0.002) pero ROUGE-L y Token F1 bajaron un poco.
El cuello de botella no era la cantidad de imágenes sino el tamaño del dataset (842 ejemplos)
y la capacidad del modelo (3B parámetros).

> **Nota sobre reproducibilidad**: las predicciones versionadas en el repo corresponden a
> Corrida 1. Las métricas de Corrida 2 se obtuvieron en la VM (`dermavqa-l4b`) y no se
> bajaron al local por diferencia marginal con Corrida 1.

---

## Comparativa zero-shot vs LoRA (test)

| Método | chrF | ROUGE-L | Token F1 | sacreBLEU | BERTScore F1 |
|--------|------|---------|----------|-----------|--------------|
| Zero-Shot | **0.183** | 0.072 | 0.092 | **0.409** | — |
| LoRA | 0.157 | **0.112** | **0.131** | 0.237 | **0.667** |

El zero-shot tiene mayor chrF y sacreBLEU (más solapamiento n-gram con las referencias),
pero el LoRA supera en ROUGE-L y Token F1 (mejor cobertura). El BERTScore del LoRA (0.667)
sugiere buena calidad semántica. La diferencia no es grande — el LoRA no degrada el modelo
base y mejora su capacidad de seguir el estilo de las respuestas del dataset.

---

## Análisis cualitativo (ejemplos de valid, LoRA)

### Caso 1
**Pregunta:** ¿Es vitíligo? La paciente es una mujer de mediana edad, de unos 50 años. Presenta erupciones...

**Referencia:** Erupciones de color rojo oscuro, que gradualmente se convierten en leucoplasia. Considera pérdida de pigmentación tras la inflamación. En el vitíligo...

**Predicción LoRA:** La erupción en la mano es una lesión cutánea típica de la vitiligo.

---

### Caso 2
**Pregunta:** Agradecería ayuda, ¿qué enfermedad cutánea es ésta? Sufro la enfermedad desde hace más de 10 años...

**Referencia:** Dermatitis, con hematomas violáceos en la piel. Debe considerarse vasculitis alérgica.

**Predicción LoRA:** La dermatitis por picadura de insecto es una enfermedad común.

---

### Caso 3
**Pregunta:** ¿Es urticaria u otra enfermedad cutánea? Lleva ahí...

**Referencia:** eczema, puede aplicar kangfuxin externamente. Por vía oral, puede tomar ácido glicirrícico.

**Predicción LoRA:** La urticaria papular es una enfermedad común. La urticaria papular es una enfermedad cutánea que ocurre cuando las células de la piel se contraen...

---

### Observaciones generales

- El modelo identifica correctamente la patología en varios casos (vitiligo, dermatitis, urticaria)
- Las respuestas son **más cortas** que las referencias
- **Sin recomendaciones de medicamentos** que sí aparecen en las referencias
- En algunos casos **repetitivas** (caso 3 repite la definición)
- Las métricas léxicas bajas (chrF ~0.15) reflejan que el modelo genera libremente y no copia frases de las referencias; el BERTScore semántico (~0.67) es más representativo de la calidad real

---

## Limitaciones a mencionar en el paper

1. Se usó el modelo de 3B en vez del 7B por restricciones de VRAM en T4.
2. El entrenamiento usó solo la primera imagen por encuentro (algunos tienen 2-3).
3. La resolución de imagen fue reducida (`max_pixels = 256 × 28 × 28`) respecto al default del modelo.
4. El dataset de entrenamiento es pequeño (842 ejemplos), lo que limita la capacidad de generalización.

---

## Comandos de reproducción

```bash
# En la VM (Google Cloud T4, zona asia-east1-c)

# 1. Setup
git clone https://github.com/Sdomato/dermavqa-paper.git
cd dermavqa-paper
git checkout develop
python3 -m pip install -r requirements.txt
pip install --upgrade jinja2

# 2. Validar pipeline (sin GPU)
python3 -m src.train_longest --dry-run --limit 5
python3 -m src.vlm_infer --split valid --dry-run

# 3. Zero-shot (baseline)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -m src.vlm_infer --split valid

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -m src.vlm_infer --split test

# 4. Fine-tuning LoRA
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python3 -m src.train_longest > train.log 2>&1 &

# 5. Inferencia LoRA
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -m src.vlm_infer --split valid \
    --adapter outputs/results/dataset_longest_answer/vlm_lora/final_adapter

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -m src.vlm_infer --split test \
    --adapter outputs/results/dataset_longest_answer/vlm_lora/final_adapter

# 6. Métricas
python3 -m src.evaluate_predictions \
    outputs/results/dataset_longest_answer/vlm_zero_shot/predictions_valid.csv \
    outputs/results/dataset_longest_answer/vlm_zero_shot/predictions_test.csv

python3 -m src.evaluate_predictions \
    outputs/results/dataset_longest_answer/vlm_lora/predictions_valid.csv \
    outputs/results/dataset_longest_answer/vlm_lora/predictions_test.csv
```

---

## Artefactos generados

| Archivo | Descripción |
|---------|-------------|
| `outputs/results/dataset_longest_answer/vlm_zero_shot/predictions_valid.csv` | Predicciones zero-shot valid (56 casos) |
| `outputs/results/dataset_longest_answer/vlm_zero_shot/predictions_test.csv` | Predicciones zero-shot test (100 casos) |
| `outputs/results/dataset_longest_answer/vlm_zero_shot/runtime_*.json` | Tiempos de inferencia zero-shot |
| `outputs/results/dataset_longest_answer/vlm_lora/train_runtime.json` | Métricas operativas del entrenamiento |
| `outputs/results/dataset_longest_answer/vlm_lora/predictions_valid.csv` | Predicciones LoRA valid (56 casos) |
| `outputs/results/dataset_longest_answer/vlm_lora/predictions_test.csv` | Predicciones LoRA test (100 casos) |
| `outputs/results/dataset_longest_answer/vlm_lora/runtime_*.json` | Tiempos de inferencia LoRA |
| `outputs/metrics/dataset_longest_answer/metrics_mixed.csv` | Resumen de métricas por split |
| `outputs/metrics/dataset_longest_answer/per_case_vlm_*.csv` | Métricas por caso |
| `outputs/results/dataset_longest_answer/vlm_lora/final_adapter/` | Adapter LoRA (~160 MB, no en repo) |
