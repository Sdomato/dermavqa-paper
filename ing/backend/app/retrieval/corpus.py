"""
Carga de la base de casos buscable.

Reutiliza las utilidades de la parte de investigación (`src/retrieval_utils.py`):
mismo dataset, misma forma de construir el texto de consulta, misma resolución
de imágenes. Así el retrieval del servicio es consistente con el de los baselines
del paper.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

# Bootstrap: agregar la raíz del repo al path para reusar src/.
# corpus.py vive en ing/backend/app/retrieval/corpus.py -> la raíz del repo
# (dermavqa-paper) está 4 niveles arriba: retrieval[0] app[1] backend[2] ing[3] root[4].
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.retrieval_utils import (  # noqa: E402  (import tras ajustar sys.path)
    build_query_text,
    clean_text,
    find_image,
    load_dataset,
)


@dataclass
class Case:
    """Un caso clínico de la base, ya normalizado para el servicio."""

    encounter_id: str
    split: str
    query_title: str
    query_content: str
    query_text: str  # título + contenido, lo que se indexa
    answer: str
    image_ids: list[str]

    @property
    def imagenes_disponibles(self) -> int:
        return sum(1 for img in self.image_ids if find_image(img) is not None)


def load_corpus(splits: str = "all") -> list[Case]:
    """
    Cargar los casos del dataset como base buscable.

    splits: "all" para los 998 casos, o una lista separada por coma de _split
            (ej. "train") para limitar la base.
    """
    records = load_dataset()

    wanted: set[str] | None = None
    if splits and splits.lower() != "all":
        wanted = {s.strip() for s in splits.split(",") if s.strip()}

    cases: list[Case] = []
    for r in records:
        split = str(r.get("_split", ""))
        if wanted is not None and split not in wanted:
            continue
        image_ids = r.get("image_ids", []) or []
        cases.append(
            Case(
                encounter_id=str(r.get("encounter_id", "")),
                split=split,
                query_title=clean_text(r.get("query_title_es", "")),
                query_content=clean_text(r.get("query_content_es", "")),
                query_text=build_query_text(r),
                answer=clean_text(r.get("answer_es", "")),
                image_ids=[str(i) for i in image_ids],
            )
        )
    return cases
