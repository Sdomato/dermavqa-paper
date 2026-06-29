# Google Cloud runbook: VLM LoRA/QLoRA reproducible

Este runbook deja reproducibles las corridas VLM con
`Qwen/Qwen2.5-VL-3B-Instruct` sobre:

- `dataset_enriched`: respuesta sintetizada/enriquecida por LLM.
- `dataset_longest_answer_by_image`: respuesta original mas larga, expandida a
  una fila por imagen.

La comparacion central del paper debe usar la misma receta para ambos:

- unidad de entrenamiento: una fila por imagen;
- splits: `train=2473`, `valid=157`, `test=314`;
- GPU: NVIDIA L4 24GB;
- QLoRA 4-bit;
- LoRA `r=16`, `alpha=32`, `dropout=0.05`;
- `batch_size=1`, `gradient_accumulation=16`;
- `epochs=1`;
- `seed=42`;
- modelo base: `Qwen/Qwen2.5-VL-3B-Instruct`.

## 1. Control de gasto

Antes de crear la VM:

- crear un budget alert de USD 50 en Google Cloud Billing;
- alertas al 50%, 80% y 90%;
- apagar la VM apenas termine el entrenamiento y se bajen resultados.

La GPU cobra mientras la VM esta encendida. Al apagarla deja de cobrar CPU/GPU,
pero el disco persistente sigue cobrando un monto menor.

## 2. Crear VM L4

Valores usados en la corrida reproducible:

```powershell
$PROJECT="nlp-derma-vqa"
$ZONE="us-central1-c"
$VM="dermavqa-vm-lora-l4"
```

Crear VM:

```powershell
gcloud.cmd compute instances create $VM `
  --project $PROJECT `
  --zone $ZONE `
  --machine-type g2-standard-4 `
  --accelerator type=nvidia-l4,count=1 `
  --maintenance-policy TERMINATE `
  --provisioning-model STANDARD `
  --boot-disk-size 150GB `
  --boot-disk-type pd-balanced `
  --image-family pytorch-2-9-cu129-ubuntu-2204-nvidia-580 `
  --image-project deeplearning-platform-release `
  --metadata install-nvidia-driver=True `
  --quiet
```

Verificar GPU:

```powershell
gcloud.cmd compute ssh $VM --zone $ZONE --project $PROJECT --command "nvidia-smi"
```

## 3. Preparar repo en la VM

```powershell
gcloud.cmd compute ssh $VM --zone $ZONE --project $PROJECT --command "mkdir -p ~/projects && if [ ! -d ~/projects/dermavqa-paper/.git ]; then git clone --branch develop https://github.com/Sdomato/dermavqa-paper.git ~/projects/dermavqa-paper; else cd ~/projects/dermavqa-paper && git fetch origin develop && git checkout develop && git pull --ff-only origin develop; fi"
```

Si estas probando cambios locales no pusheados, copiar solo los archivos
modificados. Para la version final, preferir `git pull`.

## 4. Subir datos

Las imagenes no se versionan en Git. La forma mas estable es copiar
`images_final.zip` como archivo unico y descomprimirlo en la VM:

```powershell
gcloud.cmd compute ssh $VM --zone $ZONE --project $PROJECT --command "mkdir -p /home/andyd/projects/dermavqa-paper/data/iiyi /home/andyd/projects/dermavqa-paper/outputs/datasets"

gcloud.cmd compute scp `
  "C:\Users\andyd\Udesa\NLP\prueba1\dermavqa-paper\data\iiyi\images_final.zip" `
  "${VM}:/home/andyd/projects/dermavqa-paper/data/iiyi/images_final.zip" `
  --zone $ZONE `
  --project $PROJECT `
  --quiet

gcloud.cmd compute ssh $VM --zone $ZONE --project $PROJECT --command "cd ~/projects/dermavqa-paper && python3 -m zipfile -e data/iiyi/images_final.zip data/iiyi && find data/iiyi/images_final -type f | wc -l"
```

El conteo esperado es `2945` archivos de imagen en disco, de los cuales `2944`
estan referenciados por los datasets.

## 5. Instalar dependencias VLM

La imagen PyTorch ya trae `torch` y CUDA. Instalar el stack de entrenamiento con:

```powershell
gcloud.cmd compute ssh $VM --zone $ZONE --project $PROJECT --command "cd ~/projects/dermavqa-paper && python3 -m pip install --user -U pip setuptools wheel && python3 -m pip install --user -r requirements-vlm-gcp.txt"
```

Fix obligatorio observado en la VM: `torchaudio` de la imagen puede estar
desalineado con `torch` y romper `transformers/peft`. No se usa en este proyecto.

```powershell
gcloud.cmd compute ssh $VM --zone $ZONE --project $PROJECT --command "sudo python3 -m pip uninstall -y torchaudio || true; sudo rm -rf /usr/local/lib/python3.10/dist-packages/torchaudio /usr/local/lib/python3.10/dist-packages/torchaudio-*.dist-info"
```

Chequeo de imports:

```powershell
gcloud.cmd compute ssh $VM --zone $ZONE --project $PROJECT --command "cd ~/projects/dermavqa-paper && python3 - <<'PY'
mods = ['torch','transformers','peft','trl','bitsandbytes','qwen_vl_utils','sacrebleu','bert_score','rouge_score','pandas']
for mod in mods:
    __import__(mod)
    print(mod, 'OK')
PY"
```

## 6. Datasets reproducibles

### Enriched

Artefacto esperado:

```text
outputs/datasets/dermavqa_iiyi_llm_synthesized_answer_finetune.zip
```

Validar:

```bash
python3 -m src.train_enriched --dry-run --limit 5
python3 -m src.vlm_infer_enriched --split valid --limit 5 --dry-run
```

### Longest by-image

Construir desde `dataset_longest_answer.json`:

```bash
python3 -m src.build_longest_by_image_dataset
```

Validar:

```bash
python3 -m src.train_longest_by_image --dry-run --limit 5
python3 -m src.vlm_infer_longest_by_image --split valid --limit 5 --dry-run
```

Conteos esperados para ambos datasets by-image:

| Split | Filas |
| --- | ---: |
| `train` | 2473 |
| `valid` | 157 |
| `test` | 314 |
| **Total** | **2944** |

## 7. Smoke test real

Antes de correr todo, hacer un entrenamiento mini que cargue modelo, procese
imagen y guarde adapter:

```bash
python3 -m src.train_enriched --limit 8 --epochs 1 --eval-steps 2 --save-total-limit 1
python3 -m src.train_longest_by_image --limit 8 --epochs 1 --eval-steps 2 --save-total-limit 1
```

Luego borrar esos resultados mini antes de la corrida completa:

```bash
rm -rf outputs/results/dataset_enriched/vlm_lora
rm -rf outputs/results/dataset_longest_answer/vlm_lora_by_image
```

## 8. Corridas completas en background

### Enriched

```bash
cd ~/projects/dermavqa-paper
mkdir -p outputs/results/dataset_enriched/vlm_lora
nohup bash scripts/run_enriched_vlm_lora.sh --epochs 1 > outputs/results/dataset_enriched/vlm_lora/full_run.log 2>&1 &
echo $! > outputs/results/dataset_enriched/vlm_lora/full_run.pid
```

### Longest by-image

```bash
cd ~/projects/dermavqa-paper
mkdir -p outputs/results/dataset_longest_answer/vlm_lora_by_image
nohup bash scripts/run_longest_by_image_vlm_lora.sh --epochs 1 > outputs/results/dataset_longest_answer/vlm_lora_by_image/full_run.log 2>&1 &
echo $! > outputs/results/dataset_longest_answer/vlm_lora_by_image/full_run.pid
```

Monitorear desde PowerShell:

```powershell
gcloud.cmd compute ssh $VM --zone $ZONE --project $PROJECT --command 'cd ~/projects/dermavqa-paper && tail -f outputs/results/dataset_longest_answer/vlm_lora_by_image/full_run.log'
```

Cerrar el `tail` con `Ctrl+C` no corta el entrenamiento, solo deja de mirar el
log.

## 9. Outputs esperados

Enriched:

```text
outputs/results/dataset_enriched/vlm_lora/
  final_adapter/
  checkpoints/
  training_config.json
  train_runtime.json
  train_metrics.json
  eval_metrics_valid.json
  trainer_state.json
  training_log_history.json
  training_log_history.csv
  full_run.log
  train_enriched.log
  infer_valid.log
  infer_test.log
  evaluate_predictions.log
  predictions_valid.csv
  predictions_test.csv

outputs/metrics/dataset_enriched/
  metrics_mixed.csv
  per_case_vlm_lora_valid.csv
  per_case_vlm_lora_test.csv
```

Longest by-image:

```text
outputs/results/dataset_longest_answer/vlm_lora_by_image/
  final_adapter/
  checkpoints/
  training_config.json
  train_runtime.json
  train_metrics.json
  eval_metrics_valid.json
  trainer_state.json
  training_log_history.json
  training_log_history.csv
  full_run.log
  train_longest_by_image.log
  infer_valid.log
  infer_test.log
  evaluate_predictions.log
  predictions_valid.csv
  predictions_test.csv

outputs/metrics/dataset_longest_answer/
  metrics_mixed.csv
  per_case_vlm_lora_by_image_valid.csv
  per_case_vlm_lora_by_image_test.csv
```

## 10. Que versionar y que no

Versionar en Git:

- scripts y codigo en `src/` y `scripts/`;
- `requirements-vlm-gcp.txt`;
- datasets livianos en `outputs/datasets/*.json`, `*.csv`, `*.zip`;
- metricas y predicciones CSV livianas;
- `training_config.json`, `train_runtime.json`, `train_metrics.json`,
  `eval_metrics_valid.json`, `training_log_history.csv`.

No versionar en Git:

- `data/iiyi/images_final/`;
- `data/iiyi/images_final.zip`;
- `final_adapter/`;
- `checkpoints/`;
- `.safetensors`, `.bin`, `.pt`, `.ckpt`;
- credenciales, `.env`, tokens.

## 11. Bajar resultados livianos

Ejemplo longest by-image:

```powershell
$LOCAL="C:\Users\andyd\Udesa\NLP\prueba1\dermavqa-paper"

gcloud.cmd compute scp --recurse `
  "${VM}:/home/andyd/projects/dermavqa-paper/outputs/results/dataset_longest_answer/vlm_lora_by_image" `
  "$LOCAL\outputs\results\dataset_longest_answer\" `
  --zone $ZONE `
  --project $PROJECT `
  --quiet

gcloud.cmd compute scp --recurse `
  "${VM}:/home/andyd/projects/dermavqa-paper/outputs/metrics/dataset_longest_answer" `
  "$LOCAL\outputs\metrics\" `
  --zone $ZONE `
  --project $PROJECT `
  --quiet
```

Si se desea bajar adapters/checkpoints pesados, hacerlo por `gcloud scp` o GCS,
pero mantenerlos fuera de Git.

## 12. Apagar recursos

```powershell
gcloud.cmd compute instances stop $VM --zone $ZONE --project $PROJECT
```

Si ya se bajaron resultados y no se necesita conservar el disco:

```powershell
gcloud.cmd compute instances delete $VM --zone $ZONE --project $PROJECT
```
