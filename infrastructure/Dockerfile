# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so this layer is cached across code changes.
COPY pyproject.toml ./
RUN pip install --no-cache-dir "fastapi>=0.115" "uvicorn[standard]>=0.30" \
    "httpx>=0.27" "pydantic>=2.8" "numpy>=1.26"

COPY backend ./backend
COPY data ./data
COPY ai ./ai

# Run as a non-root user.
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
