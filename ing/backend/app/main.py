"""
DermaAssist — API (Fase 1: asistente de solo retrieval).

Dada una consulta de paciente, devuelve los casos clínicos más parecidos de la
base, con su respuesta como evidencia. No genera texto nuevo: solo recupera
casos existentes, así que no puede alucinar ni inventar recomendaciones.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .retrieval.corpus import load_corpus
from .retrieval.factory import build_retriever
from .schemas import CaseHit, ConsultaRequest, ConsultaResponse

# Estado del servicio cargado al arrancar (base de casos + índice).
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cases = load_corpus(splits=settings.index_splits)
    retriever = build_retriever(settings.retriever)
    retriever.index(cases)
    _state["cases"] = cases
    _state["retriever"] = retriever
    print(f"[DermaAssist] Índice listo: {len(cases)} casos · backend={settings.retriever}")
    yield
    _state.clear()


app = FastAPI(title="DermaAssist API", version="0.1.0", lifespan=lifespan)

# CORS abierto para desarrollo (el frontend estático corre en otro origen).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "casos_indexados": len(_state.get("cases", [])),
        "retriever": settings.retriever,
    }


@app.post("/consulta", response_model=ConsultaResponse)
def consulta(req: ConsultaRequest) -> ConsultaResponse:
    cases = _state["cases"]
    retriever = _state["retriever"]

    query = f"{req.titulo} {req.contenido}".strip()
    k = req.k or settings.top_k
    hits = retriever.search(query, k=k, exclude_encounter_id=req.excluir_encounter_id)

    resultados = [
        CaseHit(
            encounter_id=cases[idx].encounter_id,
            split=cases[idx].split,
            similitud=round(score, 4),
            query_title=cases[idx].query_title,
            query_content=cases[idx].query_content,
            answer=cases[idx].answer,
            image_ids=cases[idx].image_ids,
            imagenes_disponibles=cases[idx].imagenes_disponibles,
        )
        for idx, score in hits
    ]

    return ConsultaResponse(
        consulta=query,
        retriever=settings.retriever,
        k=len(resultados),
        resultados=resultados,
    )
