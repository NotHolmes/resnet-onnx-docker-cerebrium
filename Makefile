src_dir = src
test_dir = tests
all_dirs = $(src_dir) $(test_dir)

.PHONY: build run clean-pyc clean format lint deploy

install-pipx:
	sudo apt update
	sudo apt install pipx
	pipx ensurepath

install-poetry: install-pipx
	pipx install poetry==1.8.3

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

test: format lint
	poetry run python -m tests.test

deploy: clean
	poetry export --without-hashes --format=requirements.txt --output=requirements.txt
	poetry run cerebrium deploy --config-file cerebrium.toml