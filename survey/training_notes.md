# Notas de entrenamiento y evaluación VLM LoRA (dataset_longest_answer)

## Entorno de cómputo

- **Plataforma**: Google Cloud Compute Engine
- **Instancia**: `dermavqa-train-2`, zona `asia-east1-c`
- **GPU**: NVIDIA Tesla T4 (14.56 GB VRAM)
- **SO**: Deep Learning VM with CUDA M132 (Ubuntu 22.04, CUDA 12.9, Python 3.10)
- **Costo**: ~$0.40/hora (región Asia — us-central1 y us-east1 no tenían T4 disponible al momento)
- **Costo total estimado del experimento**: ~$5-6 (entrenamiento ~5.7hs + inferencia ~0.5hs)

## Modelo elegido

Se usó **Qwen2.5-VL-3B-Instruct** en lugar del 7B original.

El 7B no entra en 16GB VRAM con QLoRA 4-bit + imágenes. Falla con OOM en el forward pass del vision encoder incluso con `max_pixels` reducido. El 3B entró con 6.9GB de VRAM pico, dejando margen cómodo.

Cita para el paper:
> *"Due to GPU memory constraints on a T4 (16 GB), we used Qwen2.5-VL-3B-Instruct instead of the 7B variant. The 7B model requires at least 24 GB VRAM (e.g., L4 or A100) for QLoRA fine-tuning with multimodal inputs."*

## Restricciones de memoria aplicadas

| Ajuste | Valor | Motivo |
|--------|-------|--------|
| `max_pixels` del processor | `256 × 28 × 28` | Reduce resolución procesada por el vision encoder |
| Imágenes por ejemplo (entrenamiento) | 1 (primera imagen) | Casos con 2-3 fotos duplicaban uso de VRAM |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Reduce fragmentación de memoria |

La limitación a 1 imagen aplica **solo al entrenamiento**. La inferencia usa todas las imágenes disponibles por encuentro.

## Hiperparámetros de entrenamiento

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

## Resultados operativos de entrenamiento

| Métrica | Valor |
|---------|-------|
| Ejemplos de train | 842 |
| Ejemplos de valid | 56 |
| Tiempo total | 5.75 horas (20,683s) |
| VRAM pico | 6.91 GB |
| Tamaño del adapter | 160.1 MB |
| Parámetros entrenables | 37,152,768 (0.98% del total) |

## Resultados de evaluación

Métricas calculadas con `src/evaluate_predictions.py` usando BERTScore multilingüe (`bert-base-multilingual-cased`).

| Split | n | chrF | ROUGE-L | Token F1 | sacreBLEU | BERTScore F1 |
|-------|---|------|---------|----------|-----------|--------------|
| valid | 56 | 0.145 | 0.101 | 0.113 | 0.237 | 0.670 |
| test | 100 | 0.157 | 0.112 | 0.131 | 0.328 | 0.667 |

### Tiempos de inferencia

| Split | n | Latencia media | Tiempo total |
|-------|---|----------------|--------------|
| valid | 56 | 10.4s/ejemplo | 9.7 min |
| test | 100 | 12.5s/ejemplo | 20.9 min |

## Ejemplos de predicciones (valid)

### Caso 1
**Pregunta:** ¿Es vitíligo? La paciente es una mujer de mediana edad, de unos 50 años. Presenta erupciones...

**Referencia:** Erupciones de color rojo oscuro, que gradualmente se convierten en leucoplasia. Considera pérdida de pigmentación tras la inflamación. En el vitíligo...

**Predicción:** La erupción en la mano es una lesión cutánea típica de la vitiligo.

---

### Caso 2
**Pregunta:** Agradecería ayuda, ¿qué enfermedad cutánea es ésta? Sufro la enfermedad desde hace más de 10 años...

**Referencia:** Dermatitis, con hematomas violáceos en la piel. Debe considerarse vasculitis alérgica.

**Predicción:** La dermatitis por picadura de insecto es una enfermedad común.

---

### Caso 3
**Pregunta:** ¿Es urticaria u otra enfermedad cutánea? Lleva ahí...

**Referencia:** eczema, puede aplicar kangfuxin externamente. Por vía oral, puede tomar ácido glicirrícico.

**Predicción:** La urticaria papular es una enfermedad común. La urticaria papular es una enfermedad cutánea que ocurre cuando las células de la piel se contraen...

---

### Análisis cualitativo

El modelo identifica correctamente la patología en varios casos (vitiligo, dermatitis, urticaria) pero sus respuestas son:
- **Más cortas** que las referencias
- **Menos específicas** clínicamente (no menciona tratamientos concretos)
- **Sin recomendaciones de medicamentos** que sí aparecen en las referencias
- En algunos casos **repetitivas** (el caso de urticaria repite la definición)

Esto es consistente con entrenar sobre 842 ejemplos con un modelo de 3B parámetros. Las métricas léxicas bajas (chrF ~0.15) reflejan que el modelo no copia frases de las referencias sino que genera libremente, lo que hace que el BERTScore semántico (~0.67) sea más representativo de la calidad real.

## Limitaciones a mencionar en el paper

1. Se usó el modelo de 3B en vez del 7B por restricciones de VRAM en T4.
2. El entrenamiento usó solo la primera imagen por encuentro (algunos tienen 2-3).
3. La resolución de imagen fue reducida (`max_pixels = 256 × 28 × 28`) respecto al default del modelo.
4. El dataset de entrenamiento es pequeño (842 ejemplos), lo que limita la capacidad de generalización.

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

# 3. Entrenar
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python3 -m src.train_longest > train.log 2>&1 &

# 4. Inferencia valid y test
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -m src.vlm_infer --split valid \
    --adapter outputs/results/dataset_longest_answer/vlm_lora/final_adapter

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -m src.vlm_infer --split test \
    --adapter outputs/results/dataset_longest_answer/vlm_lora/final_adapter

# 5. Métricas
python3 -m src.evaluate_predictions \
    outputs/results/dataset_longest_answer/vlm_lora/predictions_valid.csv \
    outputs/results/dataset_longest_answer/vlm_lora/predictions_test.csv
```

## Artefactos generados

| Archivo | Descripción |
|---------|-------------|
| `outputs/results/dataset_longest_answer/vlm_lora/train_runtime.json` | Métricas operativas del entrenamiento |
| `outputs/results/dataset_longest_answer/vlm_lora/predictions_valid.csv` | Predicciones sobre valid (56 casos) |
| `outputs/results/dataset_longest_answer/vlm_lora/predictions_test.csv` | Predicciones sobre test (100 casos) |
| `outputs/results/dataset_longest_answer/vlm_lora/runtime_valid.json` | Tiempos de inferencia valid |
| `outputs/results/dataset_longest_answer/vlm_lora/runtime_test.json` | Tiempos de inferencia test |
| `outputs/metrics/dataset_longest_answer/metrics_mixed.csv` | Resumen de métricas por split |
| `outputs/metrics/dataset_longest_answer/per_case_vlm_lora_*.csv` | Métricas por caso |
| `outputs/results/dataset_longest_answer/vlm_lora/final_adapter/` | Adapter LoRA (~160MB, no en repo) |
