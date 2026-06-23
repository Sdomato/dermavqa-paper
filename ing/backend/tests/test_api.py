"""Tests de la API de retrieval."""


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["casos_indexados"] > 0
    assert body["retriever"] in {"tfidf", "e5"}
    assert "version" in body


def test_root_redirige_a_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (307, 308)
    assert "/docs" in r.headers["location"]


def test_consulta_basica(client):
    r = client.post(
        "/consulta",
        json={"titulo": "Mancha roja en el brazo", "contenido": "Pica hace una semana", "k": 3},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["k"] == 3
    assert len(data["resultados"]) == 3
    assert data["total_casos"] > 0
    assert data["tomo_ms"] >= 0
    # Ordenados por similitud descendente.
    sims = [x["similitud"] for x in data["resultados"]]
    assert sims == sorted(sims, reverse=True)
    # Cada hit trae su respuesta como evidencia.
    assert all(x["answer"] is not None for x in data["resultados"])


def test_consulta_vacia_es_422(client):
    r = client.post("/consulta", json={"titulo": "", "contenido": "   "})
    assert r.status_code == 422


def test_k_se_acota_al_maximo(client):
    # Pedir un k absurdo no debe devolver más que max_k (default 50).
    r = client.post("/consulta", json={"titulo": "dermatitis", "k": 100000})
    assert r.status_code == 200
    assert len(r.json()["resultados"]) <= 50


def test_excluir_encounter_id(client):
    base = client.post("/consulta", json={"titulo": "psoriasis en codos", "k": 3}).json()
    excluido = base["resultados"][0]["encounter_id"]
    r = client.post(
        "/consulta", json={"titulo": "psoriasis en codos", "k": 3, "excluir_encounter_id": excluido}
    )
    ids = [x["encounter_id"] for x in r.json()["resultados"]]
    assert excluido not in ids


def test_relevancia_basica(client):
    r = client.post(
        "/consulta",
        json={"titulo": "Caída de cabello", "contenido": "se me cae el pelo y tengo zonas calvas", "k": 5},
    )
    textos = " ".join(
        (x["query_title"] + " " + x["query_content"] + " " + x["answer"]).lower()
        for x in r.json()["resultados"]
    )
    assert any(term in textos for term in ("cabello", "pelo", "calv", "alopec"))


def test_caso_por_id(client):
    hit = client.post("/consulta", json={"titulo": "acné", "k": 1}).json()["resultados"][0]
    r = client.get(f"/casos/{hit['encounter_id']}")
    assert r.status_code == 200
    assert r.json()["encounter_id"] == hit["encounter_id"]


def test_caso_inexistente_404(client):
    assert client.get("/casos/NO_EXISTE_123").status_code == 404


def test_imagen_inexistente_404(client):
    assert client.get("/imagen/no_existe.jpg").status_code == 404


def test_consulta_imagen_multipart(client):
    # Con el backend tfidf por defecto, la imagen se ignora: el test valida el
    # plumbing del endpoint multipart (form + archivo) sin deps pesadas.
    png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00"
        b"\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    r = client.post(
        "/consulta/imagen",
        data={"titulo": "lesión en la piel", "k": "3"},
        files={"imagenes": ("lesion.png", png, "image/png")},
    )
    assert r.status_code == 200
    assert r.json()["k"] == 3


def test_consulta_imagen_vacia_422(client):
    assert client.post("/consulta/imagen", data={"titulo": "", "contenido": ""}).status_code == 422
