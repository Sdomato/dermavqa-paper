"""
DermaAssist — API (Fase 1: asistente de solo retrieval).

Dada una consulta de paciente, devuelve los casos clínicos más parecidos de la
base, con su respuesta como evidencia. No genera texto nuevo: solo recupera
casos existentes, así que no puede alucinar ni inventar recomendaciones.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

from .config import APP_VERSION, settings
from .retrieval.corpus import Case, load_corpus, resolve_image
from .retrieval.factory import build_retriever
from .schemas import CaseDetail, CaseHit, ConsultaRequest, ConsultaResponse, HealthResponse

logger = logging.getLogger("dermaassist")

# Estado del servicio cargado al arrancar (base de casos + índice).
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    cases = load_corpus(splits=settings.index_splits)
    retriever = build_retriever(settings.retriever)
    retriever.index(cases)
    _state["cases"] = cases
    _state["retriever"] = retriever
    # Índice por encounter_id para lookup O(1) en /casos/{id}.
    _state["by_id"] = {c.encounter_id: c for c in cases}
    logger.info(
        "Índice listo: %d casos · backend=%s · %.1fs",
        len(cases), settings.retriever, time.perf_counter() - t0,
    )
    yield
    _state.clear()


app = FastAPI(
    title="DermaAssist API",
    version=APP_VERSION,
    summary="Recupera casos dermatológicos similares como evidencia para un médico.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=APP_VERSION,
        retriever=settings.retriever,
        casos_indexados=len(_state.get("cases", [])),
    )


@app.post("/consulta", response_model=ConsultaResponse)
def consulta(req: ConsultaRequest) -> ConsultaResponse:
    cases: list[Case] = _state["cases"]
    retriever = _state["retriever"]

    query = f"{req.titulo} {req.contenido}".strip()
    # Acotar k al rango [1, max_k] para no dejar que un cliente pida cualquier cosa.
    k = min(req.k or settings.top_k, settings.max_k)

    t0 = time.perf_counter()
    hits = retriever.search(query, k=k, exclude_encounter_id=req.excluir_encounter_id)
    tomo_ms = (time.perf_counter() - t0) * 1000

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
    logger.info("consulta k=%d -> %d hits en %.1f ms", k, len(resultados), tomo_ms)

    return ConsultaResponse(
        consulta=query,
        retriever=settings.retriever,
        k=len(resultados),
        total_casos=len(cases),
        tomo_ms=round(tomo_ms, 2),
        resultados=resultados,
    )


@app.get("/casos/{encounter_id}", response_model=CaseDetail)
def caso(encounter_id: str) -> CaseDetail:
    case: Case | None = _state.get("by_id", {}).get(encounter_id)
    if case is None:
        raise HTTPException(status_code=404, detail=f"Caso no encontrado: {encounter_id}")
    return CaseDetail(
        encounter_id=case.encounter_id,
        split=case.split,
        query_title=case.query_title,
        query_content=case.query_content,
        answer=case.answer,
        image_ids=case.image_ids,
        imagenes_disponibles=case.imagenes_disponibles,
    )


@app.get("/imagen/{image_id}")
def imagen(image_id: str) -> FileResponse:
    """Sirve la foto clínica de un caso, si está disponible en local."""
    path = resolve_image(image_id)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Imagen no disponible: {image_id}")
    return FileResponse(path)
