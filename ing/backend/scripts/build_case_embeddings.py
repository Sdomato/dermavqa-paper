"""
Genera el CACHE de embeddings de los 998 casos para el retrieval multimodal
del servicio DermaAssist (E5 texto + BiomedCLIP imagen).

Pensado para correr UNA vez en una máquina con GPU y las deps pesadas
(open_clip, torch). El servicio NO calcula esto: solo carga el .npz resultante,
así que tu compu / el server no hacen el cómputo pesado.

Reutiliza las funciones ya probadas de la parte de investigación
(`src/multimodal_retrieval.py`): mismo modelo E5, mismo BiomedCLIP, mismo
embedding por caso. La diferencia es que acá guardamos los EMBEDDINGS crudos
(no la matriz de similitud), para poder compararlos contra una query nueva.

Salida: outputs/embeddings/case_embeddings.npz
  encounter_ids : (N,)     str
  text_emb      : (N, Dt)  float32  E5, normalizado, prefijo "passage: "
  visual_emb    : (N, Dv)  float32  BiomedCLIP, media por caso, normalizado
  has_image     : (N,)     bool     si el caso tenía al menos una imagen usable
  meta          : json     model ids, dims, alpha sugerido

Uso (desde la raíz del repo, en la rama dev-ing):
  python ing/backend/scripts/build_case_embeddings.py
  python ing/backend/scripts/build_case_embeddings.py --limit 5   # prueba rápida
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Bootstrap: agregar la raíz del repo al path para reusar src/.
# scripts/build_case_embeddings.py -> raíz: scripts[0] backend[1] ing[2] root[3].
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.multimodal_retrieval import (  # noqa: E402
    BATCH_SIZE,
    E5_MODEL_ID,
    _embed_images_for_record,
    _get_embed_dim,
    _load_clip_model,
    _mean_pool,
)
from src.retrieval_utils import build_query_text, load_dataset  # noqa: E402

DEFAULT_OUT = _REPO_ROOT / "outputs" / "embeddings" / "case_embeddings.npz"


def encode_text(records: list[dict], device: torch.device) -> np.ndarray:
    """Embeddings E5 de cada caso (prefijo 'passage:', como documentos indexados)."""
    tokenizer = AutoTokenizer.from_pretrained(E5_MODEL_ID)
    model = AutoModel.from_pretrained(E5_MODEL_ID).to(device).eval()
    texts = ["passage: " + build_query_text(r) for r in records]

    out: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="E5 texto"):
        enc = tokenizer(
            texts[start : start + BATCH_SIZE],
            padding=True, truncation=True, max_length=512, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            model_out = model(**enc)
        emb = _mean_pool(model_out.last_hidden_state, enc["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        out.append(emb.cpu().numpy())
    return np.vstack(out).astype(np.float32)


def encode_visual(records: list[dict], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Embeddings BiomedCLIP de cada caso (media de sus imágenes). Reusa src/."""
    model, preprocess = _load_clip_model(device)
    dim = _get_embed_dim(model, device)
    emb = np.zeros((len(records), dim), dtype=np.float32)
    has_image = np.zeros(len(records), dtype=bool)

    for i, record in enumerate(tqdm(records, desc="BiomedCLIP imagen")):
        vec = _embed_images_for_record(record.get("image_ids") or [], model, preprocess, device)
        if vec is not None:
            emb[i] = vec
            has_image[i] = True
    return emb, has_image


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache de embeddings de casos (multimodal)")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT, help="ruta del .npz de salida")
    ap.add_argument("--limit", type=int, default=None, help="procesar solo N casos (prueba)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_dataset()
    if args.limit:
        records = records[: args.limit]
    print(f"Dispositivo: {device}  |  casos: {len(records)}")

    text_emb = encode_text(records, device)
    visual_emb, has_image = encode_visual(records, device)
    encounter_ids = np.array([str(r["encounter_id"]) for r in records])

    meta = {
        "text_model": E5_MODEL_ID,
        "text_prefix_passage": "passage: ",
        "text_prefix_query": "query: ",
        "visual_model": "BiomedCLIP (open_clip) con fallback ViT-B-32",
        "text_dim": int(text_emb.shape[1]),
        "visual_dim": int(visual_emb.shape[1]),
        "alpha_text": 0.6,
        "n": len(records),
        "n_con_imagen": int(has_image.sum()),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        encounter_ids=encounter_ids,
        text_emb=text_emb,
        visual_emb=visual_emb,
        has_image=has_image,
        meta=json.dumps(meta, ensure_ascii=False),
    )
    size_mb = args.output.stat().st_size / 1e6
    print(f"\nGuardado: {args.output}  ({size_mb:.1f} MB)")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
