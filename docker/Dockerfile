ARG PYTHON_VERSION="3.10"

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ARG POETRY_VERSION="1.8.3"
WORKDIR /root

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential git python3-pip libglib2.0-0 libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip==24.* setuptools==70.* wheel==0.43.* && \
    pip install --no-cache-dir poetry==${POETRY_VERSION} poetry-dynamic-versioning

COPY poetry.lock pyproject.toml ./
RUN poetry config virtualenvs.create false

RUN poetry install --no-dev --no-cache && \
    poetry run pip cache purge && \
    rm -rf ~/.cache/pypoetry/artifacts

COPY weights/ weights/
COPY scripts/ scripts/
RUN poetry run python3 -m scripts.convert_to_onnx
COPY src/ src/

EXPOSE 8000

RUN apt-get update && \
    apt-get install -y \
    libcudnn9-cuda-12 \
    libcudnn9-dev-cuda-12

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--workers", "1", "--log-level", "info"]