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
    casos_aprobados: int = Field(0, description="Casos sumados por el loop de mejora (Fase 4)")


class BorradorJob(BaseModel):
    """Respuesta al encolar un borrador."""

    job_id: str
    status: str = Field(..., description="pending | running | done | error")


class TerminoRiesgo(BaseModel):
    termino: str
    categoria: str


class Seguridad(BaseModel):
    """Análisis de seguridad del borrador (Fase 3)."""

    nivel: str = Field(..., description="bajo | medio | alto")
    flags: list[str] = Field(default_factory=list, description="vacio | muy_corto | repetitivo")
    diagnosticos_no_sustentados: list[str] = Field(
        default_factory=list, description="Diagnósticos del borrador ausentes en la evidencia"
    )
    recomendaciones_no_sustentadas: list[str] = Field(
        default_factory=list, description="Estudios/tratamientos del borrador ausentes en la evidencia"
    )
    cambio_de_entidad: bool = Field(
        False, description="El diagnóstico del borrador no coincide con el del caso más parecido"
    )
    banderas_rojas: list[str] = Field(
        default_factory=list, description="Señales de malignidad/urgencia en la consulta del paciente"
    )
    falsa_tranquilizacion: bool = Field(
        False, description="El borrador tranquiliza/descarta pese a haber banderas rojas en la consulta"
    )
    evidencia_debil: bool = Field(
        False, description="La similitud del caso más parecido está por debajo del umbral de confianza"
    )
    similitud_max: float | None = Field(
        None, description="Similitud del caso más parecido usado como evidencia"
    )
    terminos_riesgo: list[TerminoRiesgo] = Field(default_factory=list)


class BorradorEstado(BaseModel):
    """Estado de un borrador (poll)."""

    job_id: str
    status: str = Field(..., description="pending | running | done | error")
    evidencia: list[CaseHit] | None = Field(None, description="Casos similares usados como contexto")
    borrador: str | None = Field(None, description="Texto del borrador generado")
    seguridad: Seguridad | None = Field(None, description="Análisis de seguridad del borrador")
    error: str | None = None


ACCIONES_REVISION = {"aprobar", "editar", "rechazar"}


class RevisionRequest(BaseModel):
    """Decisión del médico sobre un borrador."""

    accion: str = Field(..., description="aprobar | editar | rechazar")
    texto_final: str | None = Field(None, description="Texto aprobado/editado (requerido si accion=editar)")
    revisor: str | None = None
    nota: str | None = None

    @model_validator(mode="after")
    def _valida(self) -> "RevisionRequest":
        if self.accion not in ACCIONES_REVISION:
            raise ValueError(f"accion inválida: {self.accion!r}. Opciones: {sorted(ACCIONES_REVISION)}")
        if self.accion == "editar" and not (self.texto_final and self.texto_final.strip()):
            raise ValueError("accion 'editar' requiere 'texto_final'")
        return self


class RevisionEntry(BaseModel):
    """Entrada del audit log."""

    id: str
    timestamp: str
    job_id: str
    accion: str
    revisor: str | None = None
    nota: str | None = None
    consulta: str | None = None
    borrador_original: str | None = None
    texto_final: str | None = None
    seguridad_nivel: str | None = None


class AuditoriaResponse(BaseModel):
    total: int
    revisiones: list[RevisionEntry]


# ── Fase 4: loop de mejora ───────────────────────────────────────────────────


class CasoAprobado(BaseModel):
    """Un caso aprobado por un médico, ya parte de la base buscable."""

    encounter_id: str
    timestamp: str
    consulta: str
    respuesta: str = Field(..., description="Respuesta validada por el médico (la evidencia)")
    revisor: str | None = None
    job_id: str | None = None
    image_ids: list[str] = Field(default_factory=list)


class DatasetAprobadosResponse(BaseModel):
    """Dataset de validación clínica humana que crece con el uso (Fase 4)."""

    total: int
    casos: list[CasoAprobado]


class MetricasResponse(BaseModel):
    """Indicadores de calidad de los borradores, derivados del audit log."""

    total_revisiones: int
    por_accion: dict[str, int] = Field(default_factory=dict)
    tasa_aprobacion: float | None = Field(
        None, description="Fracción de revisiones aprobadas o editadas (vs. rechazadas)"
    )
    similitud_borrador_final: float | None = Field(
        None, description="token-F1 medio borrador↔texto final (1.0 = aprobado sin cambios)"
    )
    edicion_media: float | None = Field(
        None, description="Cuánto reescribió el médico en promedio (1 - similitud)"
    )
    por_nivel_seguridad: dict[str, int] = Field(default_factory=dict)
