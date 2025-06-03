ARG PYTHON_VERSION="3.10"

FROM python:${PYTHON_VERSION}-slim-bullseye AS base

ARG POETRY_VERSION="1.8.3"
WORKDIR /root

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential git && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip==24.* setuptools==70.* wheel==0.43.* && \
    pip install --no-cache-dir poetry==${POETRY_VERSION} poetry-dynamic-versioning

COPY poetry.lock pyproject.toml ./
RUN poetry config virtualenvs.create false

FROM base AS builder
RUN poetry install --no-dev --no-cache && \
    poetry run pip cache purge && \
    rm -rf ~/.cache/pypoetry/artifacts

COPY weights/ weights/
COPY scripts/ scripts/
RUN poetry run python -m scripts.convert_to_onnx

FROM python:${PYTHON_VERSION}-slim-bullseye AS final

ARG PYTHON_VERSION
WORKDIR /root

COPY --from=builder /usr/local/lib/python${PYTHON_VERSION}/site-packages /usr/local/lib/python${PYTHON_VERSION}/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/models /root/models

COPY src/ src/

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--workers", "1", "--log-level", "info"]