all_dirs := src

.PHONY: build run clean-pyc clean format lint

build:
	docker compose -f docker/docker-compose.yml build

run: build
	docker compose -f docker/docker-compose.yml up

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} + || true
	find . -name '*.pyo' -exec rm -f {} + || true
	find . -name '*~' -exec rm -f {} + || true
	find . -name '__pycache__' -exec rm -fr {} + || true
	find . -name '.pytest_cache' -exec rm -fr {} + || true

clean: clean-pyc

format:
	poetry run ruff format $(all_dirs)

lint:
	poetry run ruff check $(all_dirs) --fix