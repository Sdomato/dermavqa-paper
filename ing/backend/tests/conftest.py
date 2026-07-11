import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    # El context manager dispara el lifespan (construcción del índice) una sola vez.
    with TestClient(app) as c:
        yield c
