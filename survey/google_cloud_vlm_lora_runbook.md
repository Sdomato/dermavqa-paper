# Google Cloud runbook: VLM LoRA/QLoRA sobre `dataset_enriched`

Este runbook deja reproducible la corrida de Santino para fine-tunear
`Qwen/Qwen2.5-VL-3B-Instruct` sobre el dataset enriquecido.

## Objetivo

- Dataset: `outputs/datasets/dermavqa_iiyi_llm_synthesized_answer_finetune.zip`.
- Notebook: `notebooks/03_dataset_enriched_vlm_lora.ipynb`.
- Entrada: imagen + `question_es`.
- Target: `synthesized_answer_es` si existe; en el artefacto compacto actual,
  `answer_es` enriquecida.
- Salida: `outputs/results/dataset_enriched/vlm_lora/`.

## 1. Control de gasto

Antes de crear la VM:

- Crear un budget alert de USD 50 en Google Cloud Billing.
- Configurar alertas al 50%, 80% y 90%.
- Apagar la VM apenas termine el entrenamiento.

La VM con GPU cobra mientras esta encendida. Si se apaga, deja de cobrar CPU/GPU,
pero el disco persistente sigue cobrando un monto menor.

## 2. Crear VM con L4

Opcion recomendada: crearla desde la consola de Google Cloud.

- Machine type: `g2-standard-4`.
- GPU: NVIDIA L4 24GB.
- Boot disk: 100 a 150 GB.
- Image: Deep Learning VM / PyTorch GPU.
- Region/zona: cualquiera donde haya cuota y disponibilidad de L4.

Comando CLI equivalente, ajustando `PROJECT_ID` y `ZONE`:

```bash
gcloud config set project PROJECT_ID
gcloud compute instances create dermavqa-vlm-lora-l4 \
  --zone=ZONE \
  --machine-type=g2-standard-4 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=150GB \
  --boot-disk-type=pd-balanced \
  --maintenance-policy=TERMINATE \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

Verificar GPU dentro de la VM:

```bash
nvidia-smi
```

## 3. Subir datos

Preferencia: usar un bucket GCS para poder reanudar copias.

En la maquina local:

```bash
gsutil mb gs://DERMAVQA_BUCKET
gsutil cp outputs/datasets/dermavqa_iiyi_llm_synthesized_answer_finetune.zip gs://DERMAVQA_BUCKET/
gsutil -m cp -r data/iiyi/images_final gs://DERMAVQA_BUCKET/images_final
```

En la VM:

```bash
git clone REPO_URL dermavqa-paper
cd dermavqa-paper
git checkout develop
mkdir -p outputs/datasets /mnt/disks/dermavqa/images_final
gsutil cp gs://DERMAVQA_BUCKET/dermavqa_iiyi_llm_synthesized_answer_finetune.zip outputs/datasets/
gsutil -m cp -r gs://DERMAVQA_BUCKET/images_final/* /mnt/disks/dermavqa/images_final/
```

Alternativa sin bucket:

```bash
gcloud compute ssh dermavqa-vlm-lora-l4 --zone=ZONE --command "mkdir -p ~/dermavqa-paper/outputs/datasets /mnt/disks/dermavqa/images_final"
gcloud compute scp --recurse data/iiyi/images_final dermavqa-vlm-lora-l4:/mnt/disks/dermavqa/images_final --zone=ZONE
gcloud compute scp outputs/datasets/dermavqa_iiyi_llm_synthesized_answer_finetune.zip dermavqa-vlm-lora-l4:~/dermavqa-paper/outputs/datasets/ --zone=ZONE
```

## 4. Setup en la VM

```bash
cd dermavqa-paper

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install jupyterlab

export DERMAVQA_IMAGE_ROOT="/mnt/disks/dermavqa/images_final"
```

Nota: los scripts leen `DERMAVQA_IMAGE_ROOT` para resolver imagenes. El modelo
se cambia con `--model`; el output queda fijo en
`outputs/results/dataset_enriched/vlm_lora/`.

Si el modelo pide autenticacion de Hugging Face:

```bash
huggingface-cli login
```

## 5. Ejecutar entrenamiento script-based recomendado

Primero validar dataset e imagenes sin cargar el modelo:

```bash
source .venv/bin/activate
python -m src.train_enriched --dry-run --limit 5
python -m src.vlm_infer_enriched --split valid --limit 5 --dry-run
```

Para una corrida completa que deje entrenamiento, predicciones y metricas:

```bash
bash scripts/run_enriched_vlm_lora.sh --epochs 1
```

Ese comando ejecuta:

1. `src.train_enriched`: entrena QLoRA y guarda adapter, checkpoints, estado y metricas train/valid.
2. `src.vlm_infer_enriched --split valid`: genera `predictions_valid.csv`.
3. `src.vlm_infer_enriched --split test`: genera `predictions_test.csv`.
4. `src.evaluate_predictions`: calcula metricas automaticas valid/test.

### Corrida realizada

La corrida de `dataset_enriched` ya fue ejecutada con `--epochs 1` en una VM
Google Cloud con NVIDIA L4. Resultado operativo:

- `n_train`: 2473 filas por imagen.
- `n_eval`: 157 filas por imagen.
- tiempo de entrenamiento: 4636.4 s (77.3 min).
- VRAM pico: 6.73 GB.
- adapter final: 160.1 MB.
- inferencia valid: 157 filas, 15.96 s/ejemplo.
- inferencia test: 314 filas, 14.97 s/ejemplo.
- metricas finales: `outputs/metrics/dataset_enriched/metrics_mixed.csv`.

Si despues queremos la corrida de 3 epochs para comparar con "dataset completo x3":

```bash
bash scripts/run_enriched_vlm_lora.sh --epochs 3
```

## 6. Outputs generados/esperados

```text
outputs/results/dataset_enriched/vlm_lora/
  final_adapter/                 # pesos LoRA finales/best checkpoint
  checkpoints/                   # checkpoints LoRA intermedios
  training_config.json
  train_runtime.json
  train_metrics.json
  eval_metrics_valid.json        # eval_loss de valid durante entrenamiento
  trainer_state.json
  training_log_history.json
  training_log_history.csv       # loss/eval_loss por step
  train_enriched.log
  infer_valid.log
  infer_test.log
  evaluate_predictions.log
  predictions_valid.csv
  predictions_test.csv

outputs/metrics/dataset_enriched/
  metrics_mixed.csv              # resumen valid + test
  per_case_vlm_lora_valid.csv
  per_case_vlm_lora_test.csv
```

Copiar resultados al bucket:

```bash
gsutil -m cp -r outputs/results/dataset_enriched/vlm_lora gs://DERMAVQA_BUCKET/outputs/results/dataset_enriched/
gsutil -m cp -r outputs/metrics/dataset_enriched gs://DERMAVQA_BUCKET/outputs/metrics/
```

## 7. Apagar recursos

```bash
gcloud compute instances stop dermavqa-vlm-lora-l4 --zone=ZONE
```

Si ya se bajaron resultados y no se necesita conservar el disco:

```bash
gcloud compute instances delete dermavqa-vlm-lora-l4 --zone=ZONE
```

## Fallback si aparece OOM

Aplicar en este orden:

1. Confirmar primero `python -m src.train_enriched --dry-run --limit 5`.
2. Mantener `--batch-size 1` y subir `--grad-accum` si se necesita compensar.
3. Bajar `--limit` para hacer smoke tests mas chicos.
4. Si el OOM ocurre durante vision, reducir el `max_pixels` hardcodeado en
   `src/train_enriched.py` / `src/vlm_infer_enriched.py`.
5. Si persiste, pasar a una GPU mayor o reducir el modelo con `--model`.
