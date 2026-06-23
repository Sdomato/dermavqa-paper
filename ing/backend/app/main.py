"""
DermaAssist — API (Fase 1: asistente de solo retrieval).

Dada una consulta de paciente, devuelve los casos clínicos más parecidos de la
base, con su respuesta como evidencia. No genera texto nuevo: solo recupera
casos existentes, así que no puede alucinar ni inventar recomendaciones.
"""

import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

from .config import APP_VERSION, settings
from .generation.factory import build_generator
from .jobs import JobStore
from .retrieval.corpus import Case, load_corpus, resolve_image
from .retrieval.factory import build_retriever
from .safety.analyzer import analizar
from .schemas import (
    BorradorEstado,
    BorradorJob,
    CaseDetail,
    CaseHit,
    ConsultaRequest,
    ConsultaResponse,
    HealthResponse,
)

logger = logging.getLogger("dermaassist")

# Estado del servicio cargado al arrancar (base de casos + índice).
_state: dict = {}
_jobs = JobStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    cases = load_corpus(splits=settings.index_splits)
    retriever = build_retriever(settings.retriever)
    retriever.index(cases)
    _state["cases"] = cases
    _state["retriever"] = retriever
    _state["generator"] = build_generator(settings.generator)
    # Índice por encounter_id para lookup O(1) en /casos/{id}.
    _state["by_id"] = {c.encounter_id: c for c in cases}
    logger.info(
        "Índice listo: %d casos · retriever=%s · generator=%s · %.1fs",
        len(cases), settings.retriever, settings.generator, time.perf_counter() - t0,
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
        generator=settings.generator,
        casos_indexados=len(_state.get("cases", [])),
    )


def _to_casehits(hits: list[tuple[int, float]]) -> list[CaseHit]:
    cases: list[Case] = _state["cases"]
    return [
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


def _guardar_uploads(imagenes: list[UploadFile]) -> list[Path]:
    """Guarda los archivos subidos en temporales y devuelve sus rutas."""
    paths: list[Path] = []
    for up in imagenes:
        if not up.filename:
            continue
        suffix = Path(up.filename).suffix or ".img"
        fd, tmp = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(up.file.read())
        paths.append(Path(tmp))
    return paths


def _responder(query: str, k: int, exclude: str | None, image_paths: list[Path] | None) -> ConsultaResponse:
    """Corre la búsqueda y arma la respuesta. Compartido por /consulta y /consulta/imagen."""
    retriever = _state["retriever"]
    k = min(k, settings.max_k)

    t0 = time.perf_counter()
    hits = retriever.search(query, k=k, exclude_encounter_id=exclude, query_image_paths=image_paths)
    tomo_ms = (time.perf_counter() - t0) * 1000

    resultados = _to_casehits(hits)
    logger.info("consulta k=%d img=%s -> %d hits en %.1f ms",
                k, bool(image_paths), len(resultados), tomo_ms)

    return ConsultaResponse(
        consulta=query,
        retriever=settings.retriever,
        k=len(resultados),
        total_casos=len(_state["cases"]),
        tomo_ms=round(tomo_ms, 2),
        resultados=resultados,
    )


@app.post("/consulta", response_model=ConsultaResponse)
def consulta(req: ConsultaRequest) -> ConsultaResponse:
    """Consulta solo-texto (JSON). Funciona con cualquier backend."""
    query = f"{req.titulo} {req.contenido}".strip()
    return _responder(query, req.k or settings.top_k, req.excluir_encounter_id, None)


@app.post("/consulta/imagen", response_model=ConsultaResponse)
def consulta_imagen(
    titulo: str = Form(""),
    contenido: str = Form(""),
    k: int | None = Form(None),
    excluir_encounter_id: str | None = Form(None),
    imagenes: list[UploadFile] = File(default=[]),
) -> ConsultaResponse:
    """
    Consulta con imágenes (multipart). El backend multimodal las usa para fusionar
    señal visual; los backends solo-texto las ignoran.
    """
    query = f"{titulo} {contenido}".strip()
    if not query:
        raise HTTPException(
            status_code=422,
            detail="La consulta no puede estar vacía: completá 'titulo' o 'contenido'.",
        )

    tmp_paths = _guardar_uploads(imagenes)
    try:
        return _responder(query, k or settings.top_k, excluir_encounter_id, tmp_paths or None)
    finally:
        for p in tmp_paths:
            try:
                p.unlink()
            except OSError:
                pass


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


# ── Fase 2: borrador con RAG (asíncrono por la latencia del VLM) ─────────────────

def _borrador_task(query: str, k: int, exclude: str | None, tmp_paths: list[Path]) -> dict:
    """Recupera evidencia, arma el prompt RAG y genera el borrador. Limpia los temporales."""
    try:
        retriever = _state["retriever"]
        generator = _state["generator"]
        hits = retriever.search(
            query, k=min(k, settings.max_k), exclude_encounter_id=exclude,
            query_image_paths=tmp_paths or None,
        )
        evidencia = _to_casehits(hits)
        evidence_dicts = [
            {"answer": e.answer, "similitud": e.similitud, "query_title": e.query_title}
            for e in evidencia
        ]
        borrador = generator.generate(query, evidence_dicts, tmp_paths or None)
        seguridad = analizar(borrador, [e.answer for e in evidencia])
        return {"evidencia": evidencia, "borrador": borrador, "seguridad": seguridad}
    finally:
        for p in tmp_paths:
            try:
                p.unlink()
            except OSError:
                pass


@app.post("/borrador", response_model=BorradorJob)
def borrador(
    titulo: str = Form(""),
    contenido: str = Form(""),
    k: int | None = Form(None),
    excluir_encounter_id: str | None = Form(None),
    imagenes: list[UploadFile] = File(default=[]),
) -> BorradorJob:
    """
    Encola la generación de un borrador (RAG: recupera casos similares y genera).
    Devuelve un job_id; el cliente hace poll de GET /borrador/{job_id}.
    Es asíncrono porque el VLM tarda 12–26 s.
    """
    query = f"{titulo} {contenido}".strip()
    if not query:
        raise HTTPException(
            status_code=422,
            detail="La consulta no puede estar vacía: completá 'titulo' o 'contenido'.",
        )
    tmp_paths = _guardar_uploads(imagenes)
    kk = k or settings.rag_k
    job_id = _jobs.submit(lambda: _borrador_task(query, kk, excluir_encounter_id, tmp_paths))
    return BorradorJob(job_id=job_id, status="pending")


@app.get("/borrador/{job_id}", response_model=BorradorEstado)
def borrador_estado(job_id: str) -> BorradorEstado:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job no encontrado: {job_id}")
    result = job.result or {}
    return BorradorEstado(
        job_id=job.id,
        status=job.status,
        evidencia=result.get("evidencia"),
        borrador=result.get("borrador"),
        seguridad=result.get("seguridad"),
        error=job.error,
    )
