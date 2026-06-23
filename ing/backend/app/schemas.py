"""
Contratos de la API (request / response).

Define el "qué entra y qué sale" del endpoint principal. Es el contrato que
el frontend y cualquier cliente pueden asumir estable.
"""

from pydantic import BaseModel, Field, model_validator


class ConsultaRequest(BaseModel):
    """Lo que envía el médico: la consulta del paciente."""

    titulo: str = Field("", description="Título / motivo de consulta")
    contenido: str = Field("", description="Descripción del caso en español")
    k: int | None = Field(None, ge=1, description="Cuántos casos similares devolver (default del server)")
    excluir_encounter_id: str | None = Field(
        None, description="Excluir este caso de los resultados (evita recuperarse a sí mismo)"
    )

    @model_validator(mode="after")
    def _query_no_vacia(self) -> "ConsultaRequest":
        if not (self.titulo.strip() or self.contenido.strip()):
            raise ValueError("La consulta no puede estar vacía: completá 'titulo' o 'contenido'.")
        return self


class CaseHit(BaseModel):
    """Un caso similar recuperado, con su respuesta como evidencia."""

    encounter_id: str
    split: str
    similitud: float = Field(..., description="Score de similitud [0,1], mayor = más parecido")
    query_title: str
    query_content: str
    answer: str = Field(..., description="Respuesta del caso recuperado (la evidencia)")
    image_ids: list[str]
    imagenes_disponibles: int = Field(..., description="Cuántas de esas imágenes existen en local")


class ConsultaResponse(BaseModel):
    consulta: str
    retriever: str
    k: int
    total_casos: int = Field(..., description="Tamaño de la base buscada")
    tomo_ms: float = Field(..., description="Tiempo de la búsqueda en milisegundos")
    resultados: list[CaseHit]


class CaseDetail(BaseModel):
    """Detalle completo de un caso de la base (endpoint /casos/{id})."""

    encounter_id: str
    split: str
    query_title: str
    query_content: str
    answer: str
    image_ids: list[str]
    imagenes_disponibles: int


class HealthResponse(BaseModel):
    status: str
    version: str
    retriever: str
    generator: str
    casos_indexados: int


class BorradorJob(BaseModel):
    """Respuesta al encolar un borrador."""

    job_id: str
    status: str = Field(..., description="pending | running | done | error")


class BorradorEstado(BaseModel):
    """Estado de un borrador (poll)."""

    job_id: str
    status: str = Field(..., description="pending | running | done | error")
    evidencia: list[CaseHit] | None = Field(None, description="Casos similares usados como contexto")
    borrador: str | None = Field(None, description="Texto del borrador generado")
    error: str | None = None
