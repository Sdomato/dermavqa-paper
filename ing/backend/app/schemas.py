"""
Contratos de la API (request / response).

Define el "qué entra y qué sale" del endpoint principal. Es el contrato que
el frontend y cualquier cliente pueden asumir estable.
"""

from pydantic import BaseModel, Field


class ConsultaRequest(BaseModel):
    """Lo que envía el médico: la consulta del paciente."""

    titulo: str = Field("", description="Título / motivo de consulta")
    contenido: str = Field("", description="Descripción del caso en español")
    k: int | None = Field(None, description="Cuántos casos similares devolver (default del server)")
    excluir_encounter_id: str | None = Field(
        None, description="Excluir este caso de los resultados (evita recuperarse a sí mismo)"
    )


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
    resultados: list[CaseHit]
