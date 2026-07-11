# Atajos de desarrollo del backend. Correr desde ing/backend/.

.PHONY: install run test lint fmt

install:
	pip install -r requirements.txt

run:
	uvicorn app.main:app --reload

test:
	python -m pytest -q

lint:
	ruff check app tests

fmt:
	ruff format app tests
