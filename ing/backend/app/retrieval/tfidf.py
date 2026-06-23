"""
Backend de retrieval TF-IDF.

Liviano y sin descargas: ideal para desarrollo local y demos. Usa el mismo
enfoque que `src/tfidf_retrieval.py` del paper (TF-IDF sobre el texto de la
consulta), pero adaptado a indexar una base y consultar con textos nuevos.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base import Retriever
from .corpus import Case


class TfidfRetriever(Retriever):
    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer()
        self._matrix = None
        self._encounter_ids: list[str] = []

    def index(self, cases: list[Case]) -> None:
        corpus = [c.query_text for c in cases]
        self._matrix = self._vectorizer.fit_transform(corpus)
        self._encounter_ids = [c.encounter_id for c in cases]

    def search(
        self,
        query: str,
        k: int,
        exclude_encounter_id: str | None = None,
        query_image_paths: list | None = None,  # ignorado: backend solo-texto
    ) -> list[tuple[int, float]]:
        if self._matrix is None:
            raise RuntimeError("El índice no fue construido (llamar index() primero)")

        query_vec = self._vectorizer.transform([query])
        # Similitud coseno contra todos los casos; TF-IDF ya da scores en [0, 1].
        scores = cosine_similarity(query_vec, self._matrix)[0]

        order = scores.argsort()[::-1]  # mayor a menor
        hits: list[tuple[int, float]] = []
        for idx in order:
            if exclude_encounter_id and self._encounter_ids[idx] == exclude_encounter_id:
                continue
            hits.append((int(idx), float(scores[idx])))
            if len(hits) >= k:
                break
        return hits
