"""Tests básicos de humo de la API."""

from fastapi.testclient import TestClient

from app.main import app


def test_health_y_consulta():
    # El lifespan (carga del índice) corre dentro del context manager del client.
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["status"] == "ok"
        assert body["casos_indexados"] > 0

        resp = client.post(
            "/consulta",
            json={"titulo": "Mancha roja en el brazo", "contenido": "Pica desde hace una semana", "k": 3},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["k"] == 3
        assert len(data["resultados"]) == 3
        # Los resultados vienen ordenados por similitud descendente.
        sims = [r["similitud"] for r in data["resultados"]]
        assert sims == sorted(sims, reverse=True)
        # Cada hit trae su respuesta como evidencia.
        assert all("answer" in r for r in data["resultados"])
