# Notas de entrenamiento VLM (dataset_longest_answer)

## Entorno

- **Plataforma**: Google Cloud Compute Engine
- **Instancia**: `dermavqa-train-2`, zona `asia-east1-c`
- **GPU**: NVIDIA Tesla T4 (14.56 GB VRAM)
- **SO**: Deep Learning VM with CUDA M132 (Ubuntu 22.04, CUDA 12.9, Python 3.10)
- **Costo estimado**: ~$0.40/hora (región Asia)

## Modelo

Se usó **Qwen2.5-VL-3B-Instruct** en lugar del 7B original porque el 7B no entra en 16GB VRAM ni con QLoRA 4-bit. El 3B permite entrenamiento con margen razonable.

Limitación a documentar en el paper:
> *"Due to GPU memory constraints on a T4 (16 GB), we used Qwen2.5-VL-3B-Instruct instead of the 7B variant. The 7B model requires at least 24 GB VRAM for QLoRA fine-tuning with multimodal inputs."*

## Restricciones de memoria aplicadas

Para que el entrenamiento entre en 16GB se aplicaron tres ajustes:

| Ajuste | Valor | Motivo |
|--------|-------|--------|
| `max_pixels` del processor | `256 * 28 * 28` | Reduce resolución de imagen procesada por el vision encoder |
| Imágenes por ejemplo | 1 (primera imagen) | Algunos encuentros tienen 2-3 fotos; usar todas dobla la memoria del vision encoder |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Reduce fragmentación de VRAM |

La limitación a 1 imagen aplica **solo al entrenamiento**. La inferencia (`vlm_infer.py`) usa todas las imágenes disponibles por encuentro.

## Comando de entrenamiento (prueba con 50 ejemplos)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -m src.train_longest --epochs 1 --limit 50 --grad-accum 32
```

## Comando de entrenamiento completo (3 épocas)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python3 -m src.train_longest > train.log 2>&1 &
echo $! > train.pid
tail -f train.log
```

## Hiperparámetros finales

| Parámetro | Valor |
|-----------|-------|
| Modelo | Qwen2.5-VL-3B-Instruct |
| Épocas | 3 |
| LR | 2e-4 |
| Batch size | 1 |
| Grad accumulation | 16 (prueba: 32) |
| LoRA r | 16 |
| LoRA alpha | 32 |
| Cuantización | QLoRA 4-bit (nf4, bfloat16) |
| max_pixels | 256 × 28 × 28 |
| Imágenes por ejemplo | 1 (entrenamiento) / todas (inferencia) |

## Setup en la VM

```bash
# Clonar repo
git clone https://damiandistefano:<TOKEN>@github.com/Sdomato/dermavqa-paper.git
cd dermavqa-paper
git checkout develop

# Instalar dependencias
python3 -m pip install -r requirements.txt
pip install --upgrade jinja2  # versión mínima 3.1.0 requerida por apply_chat_template

# Subir imágenes desde Mac (ejecutar en terminal local)
gcloud compute scp --recurse /ruta/local/imagenes/ dermavqa-train-2:~/dermavqa-paper/data/images/ --zone asia-east1-c

# Validar antes de entrenar
python3 -m src.train_longest --dry-run --limit 5
# Debe mostrar N > 0 ejemplos usables

# Lanzar entrenamiento
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python3 -m src.train_longest > train.log 2>&1 &
echo $! > train.pid
tail -f train.log
```

## Apagar la instancia al terminar

```bash
# Bajar resultados (desde Mac)
gcloud compute scp --recurse dermavqa-train-2:~/dermavqa-paper/outputs/ ./outputs/ --zone asia-east1-c

# Apagar (la instancia sigue cobrando si queda prendida)
gcloud compute instances stop dermavqa-train-2 --zone asia-east1-c
```
