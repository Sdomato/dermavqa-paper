# Imagen del backend de DermaAssist (Fase 1: retrieval).
# El contexto de build es la RAÍZ del repo, porque el backend reutiliza
# src/retrieval_utils.py y el dataset de outputs/.
FROM python:3.11-slim

WORKDIR /repo

# Dependencias del backend.
COPY ing/backend/requirements.txt ing/backend/requirements.txt
RUN pip install --no-cache-dir -r ing/backend/requirements.txt

# Código de investigación reutilizado + dataset + backend.
COPY src/ src/
COPY outputs/datasets/ outputs/datasets/
COPY ing/backend/ ing/backend/

WORKDIR /repo/ing/backend
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
