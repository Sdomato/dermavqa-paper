"""
Tests unitarios de robustez: validación de config, factories, cola de jobs,
errores de los retrievers y utilidades del corpus. No requieren modelos pesados.
"""

import time
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.generation.factory import build_generator
from app.generation.stub import StubGenerator
from app.generation.vlm import VLMGenerator
from app.jobs import JobStore
from app.retrieval.corpus import Case, load_corpus, resolve_image
from app.retrieval.e5 import E5Retriever
from app.retrieval.factory import build_retriever
from app.retrieval.multimodal import MultimodalRetriever
from app.retrieval.tfidf import TfidfRetriever

# ── config: validación fail-fast ────────────────────────────────────────────

def test_config_retriever_invalido():
    with pytest.raises(ValueError, match="DERMA_RETRIEVER"):
        Settings(retriever="inexistente")


def test_config_generator_invalido():
    with pytest.raises(ValueError, match="DERMA_GENERATOR"):
        Settings(generator="inexistente")


def test_config_top_k_invalido():
    with pytest.raises(ValueError):
        Settings(top_k=0)


def test_config_max_k_menor_que_top_k():
    with pytest.raises(ValueError):
        Settings(top_k=10, max_k=5)


def test_config_valida_ok():
    s = Settings(retriever="multimodal", generator="vlm")
    assert s.retriever == "multimodal" and s.generator == "vlm"


# ── factories ───────────────────────────────────────────────────────────────

def test_retrieval_factory_construye_cada_backend():
    assert isinstance(build_retriever("tfidf"), TfidfRetriever)
    assert isinstance(build_retriever("e5"), E5Retriever)
    assert isinstance(build_retriever("multimodal"), MultimodalRetriever)


def test_retrieval_factory_desconocido():
    with pytest.raises(ValueError, match="desconocido"):
        build_retriever("loquesea")


def test_generation_factory_construye():
    assert isinstance(build_generator("stub"), StubGenerator)
    assert isinstance(build_generator("vlm"), VLMGenerator)


def test_generation_factory_desconocido():
    with pytest.raises(ValueError, match="desconocido"):
        build_generator("loquesea")


def test_vlm_generator_construye_sin_cargar_modelo():
    # Construir no debe cargar torch/Qwen (es lazy en _ensure_model).
    gen = VLMGenerator(adapter=None, max_new_tokens=128)
    assert gen.adapter is None and gen.max_new_tokens == 128 and gen._model is None


# ── cola de jobs ────────────────────────────────────────────────────────────

def _wait(store, job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = store.get(job_id)
        if job and job.status in ("done", "error"):
            return job
        time.sleep(0.02)
    raise AssertionError("job no terminó")


def test_job_exitoso():
    store = JobStore()
    job_id = store.submit(lambda: 21 * 2)
    job = _wait(store, job_id)
    assert job.status == "done" and job.result == 42 and job.error is None


def test_job_con_error():
    store = JobStore()

    def explota():
        raise RuntimeError("boom")

    job_id = store.submit(explota)
    job = _wait(store, job_id)
    assert job.status == "error" and "boom" in job.error


def test_job_inexistente():
    assert JobStore().get("nope") is None


# ── retrievers: errores y branches ──────────────────────────────────────────

def test_tfidf_buscar_sin_indexar():
    with pytest.raises(RuntimeError):
        TfidfRetriever().search("hola", k=1)


def test_multimodal_buscar_sin_indexar():
    r = MultimodalRetriever("/no/importa.npz")
    with pytest.raises(RuntimeError):
        r._search_with_vectors(np.zeros(768, dtype=np.float32), None, k=1)


def test_multimodal_caso_faltante_en_cache():
    cache = Path(Settings().embeddings_path)
    if not cache.exists():
        pytest.skip("falta el cache de embeddings")
    fake = Case(
        encounter_id="NO_EXISTE_999", split="train", query_title="x",
        query_content="y", query_text="x y", answer="z", image_ids=[],
    )
    r = MultimodalRetriever(cache)
    r.index([fake])  # no debe romper; el caso queda con vector cero
    assert r._has_image[0] == False  # noqa: E712 — chequeo explícito de numpy bool
    assert float(np.linalg.norm(r._text_emb[0])) == 0.0


# ── corpus ──────────────────────────────────────────────────────────────────

def test_corpus_filtro_de_split():
    todos = load_corpus(splits="all")
    train = load_corpus(splits="train")
    assert len(todos) == 998
    assert 0 < len(train) < len(todos)
    assert all(c.split == "train" for c in train)


def test_resolve_image():
    # El caso negativo siempre vale (no depende de tener imágenes en disco).
    assert resolve_image("definitivamente_no_existe.jpg") is None
    # El positivo solo si las imágenes están en local (no es el caso en CI).
    cases = load_corpus(splits="all")
    con_img = next(c for c in cases if c.image_ids)
    path = resolve_image(con_img.image_ids[0])
    if path is None:
        pytest.skip("imágenes no presentes en local (ej. CI)")
    assert path.exists()
