.PHONY: install test lint fmt run bench quickstart docker-build docker-up clean

install:
	pip install -e ".[dev]"

test:
	pytest -v

lint:
	ruff check .

fmt:
	ruff check . --fix

run:
	uvicorn backend.main:app --reload

bench:
	python -m benchmarks.latency_bench

quickstart:
	python -m examples.quickstart

docker-build:
	docker build -f infrastructure/Dockerfile -t ai-systems-lab .

docker-up:
	docker compose -f infrastructure/docker-compose.yml up --build

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
