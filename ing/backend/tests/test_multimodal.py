"""
Tests del retrieval multimodal.

La fusión y el núcleo de búsqueda son numpy puro: se testean sin modelos.
El test contra el cache real inyecta el embedding de un caso como si fuera la
query y verifica que el caso se recupera a sí mismo (valida carga + alineación
+ scoring). No descarga ni corre E5/BiomedCLIP.
"""

from pathlib import Path

import numpy as np
import pytest

from app.config import settings
from app.retrieval.corpus import load_corpus
from app.retrieval.multimodal import MultimodalRetriever, fuse_scores

CACHE = Path(settings.embeddings_path)
_have_cache = CACHE.exists()


# ── fusión (sintético) ──────────────────────────────────────────────────────

def test_fuse_solo_texto():
    text = np.array([0.1, 0.9, 0.5])
    has_image = np.array([True, True, True])
    out = fuse_scores(text, None, alpha=0.6, has_image=has_image)
    # min-max: el mayor va a 1, el menor a 0.
    assert out.argmax() == 1
    assert out.min() == 0.0 and out.max() == 1.0


def test_fuse_combina_texto_y_visual():
    text = np.array([1.0, 0.0])      # caso 0 gana por texto
    visual = np.array([0.0, 1.0])    # caso 1 gana por visual
    has_image = np.array([True, True])
    out = fuse_scores(text, visual, alpha=0.6, has_image=has_image)
    # alpha=0.6 favorece texto -> caso 0 debe quedar arriba.
    assert out[0] > out[1]


def test_fuse_caso_sin_imagen_no_matchea_visual():
    text = np.array([0.0, 0.0, 0.0])      # sin señal de texto
    visual = np.array([1.0, 0.5, 0.9])    # caso 2 tendría buen match visual...
    has_image = np.array([True, True, False])  # ...pero no tiene imagen
    out = fuse_scores(text, visual, alpha=0.6, has_image=has_image)
    # El caso 2 (sin imagen) recibe -inf visual -> no puede ganar por la vía visual.
    assert out[0] > out[2]
    assert out[2] == 0.0


# ── núcleo de retrieval contra el cache real ────────────────────────────────

@pytest.mark.skipif(not _have_cache, reason="falta outputs/embeddings/case_embeddings.npz")
def test_recupera_self_por_texto():
    cases = load_corpus(splits="all")
    r = MultimodalRetriever(CACHE, alpha=0.6)
    r.index(cases)
    i = 100
    q_text = r._text_emb[i]  # el propio embedding de texto del caso i
    hits = r._search_with_vectors(q_text, None, k=1)
    assert hits[0][0] == i  # el caso i se recupera a sí mismo


@pytest.mark.skipif(not _have_cache, reason="falta outputs/embeddings/case_embeddings.npz")
def test_recupera_self_multimodal():
    cases = load_corpus(splits="all")
    r = MultimodalRetriever(CACHE, alpha=0.6)
    r.index(cases)
    i = 250
    hits = r._search_with_vectors(r._text_emb[i], r._visual_emb[i], k=3)
    assert hits[0][0] == i
    # scores ordenados desc y en rango fusión [0, 1].
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
    assert -1e-6 <= scores[0] <= 1.0 + 1e-6  # rango fusión [0,1] (tolerancia float32)


@pytest.mark.skipif(not _have_cache, reason="falta outputs/embeddings/case_embeddings.npz")
def test_excluir_self():
    cases = load_corpus(splits="all")
    r = MultimodalRetriever(CACHE, alpha=0.6)
    r.index(cases)
    i = 7
    eid = cases[i].encounter_id
    hits = r._search_with_vectors(r._text_emb[i], None, k=3, exclude_encounter_id=eid)
    assert all(cases[idx].encounter_id != eid for idx, _ in hits)


def test_cache_inexistente_explica():
    r = MultimodalRetriever("/no/existe/embeddings.npz")
    with pytest.raises(FileNotFoundError):
        r.index([])
