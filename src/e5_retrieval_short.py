"""
Baseline de Retrieval Textual con Multilingual E5 sobre dataset_short_answer.

Modelo: intfloat/multilingual-e5-base
Salida: outputs/results/dataset_short_answer/retrieval_textual_e5/e5_results.json
"""

from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.retrieval_utils import (
    PROJECT_ROOT,
    build_query_text,
    build_results,
    clean_text,
    load_dataset,
    save_results,
    top1_excluding_self,
)

MODEL_ID = "intfloat/multilingual-e5-base"
BATCH_SIZE = 64
DATASET_PATH = PROJECT_ROOT / "outputs" / "datasets" / "dataset_short_answer.json"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "results"
    / "dataset_short_answer"
    / "retrieval_textual_e5"
    / "e5_results.json"
)


def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def encode_texts(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
) -> np.ndarray:
    all_embeddings: list[np.ndarray] = []
    model.eval()
    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Encoding E5"):
        batch = ["query: " + t for t in texts[start : start + BATCH_SIZE]]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            output = model(**encoded)
        emb = mean_pool(output.last_hidden_state, encoded["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        all_embeddings.append(emb.cpu().numpy())
    return np.vstack(all_embeddings)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    records = load_dataset(DATASET_PATH)
    print(f"Registros cargados: {len(records)}")

    texts = [build_query_text(r) for r in records]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device)

    embeddings = encode_texts(texts, tokenizer, model, device)
    sim_matrix: np.ndarray = embeddings @ embeddings.T
    best_idx, best_scores = top1_excluding_self(sim_matrix)

    results = build_results(records, best_idx, best_scores)
    for i, r in enumerate(results):
        r["retrieved_short_answer_es"] = clean_text(
            records[int(best_idx[i])].get("answer_es", "")
        )
        r.pop("retrieved_answer_es", None)

    save_results(results, OUTPUT_PATH)


if __name__ == "__main__":
    main()
