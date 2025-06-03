# ML Inference API

This repository provides a FastAPI-based machine learning inference API for image classification using an ONNX model. It includes scripts for model export, local and remote testing, Docker deployment, and CI integration.

## Table of Contents
- [ML Inference API](#ml-inference-api)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Setup](#setup)
  - [Exporting the Model to ONNX](#exporting-the-model-to-onnx)
  - [Local Model Inference](#local-model-inference)
  - [Running the API Locally](#running-the-api-locally)
  - [Testing the API Locally](#testing-the-api-locally)
  - [Docker Deployment](#docker-deployment)
  - [Testing the Deployed API (Remote)](#testing-the-deployed-api-remote)
  - [Makefile Commands](#makefile-commands)
  - [Troubleshooting](#troubleshooting)
  - [API Usage Example](#api-usage-example)

---

## Project Structure

```
resnet-onnx-docker-cerebrium/
├── src/                # API and model code
├── scripts/            # Model export and utility scripts
├── images/             # Sample images for testing
├── models/             # Exported ONNX model
├── tests/              # Test scripts
├── docker/             # Dockerfile and docker-compose
├── Makefile            # Common commands
├── pyproject.toml      # Python/Poetry config
├── requirements.txt    # Exported requirements
├── cerebrium.toml      # Cerebrium deployment config
└── README.md           # This file
```

---

## Setup

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd resnet-onnx-docker-cerebrium
   ```

2. **Install Poetry (if not already installed):**
   - You can install Poetry manually:
     ```sh
     pip install --user poetry==1.8.3
     ```
   - Or, if you have `make` installed, you can use the provided Makefile target:
     ```sh
     make install-poetry
     ```
     > **Note:** If you don't have `make`, install it first (e.g., `sudo apt install make` on Ubuntu/Debian).

3. **Install dependencies:**
   ```sh
   poetry install
   ```

---

## Exporting the Model to ONNX

The ONNX model is required for inference. If not already present in `models/model.onnx`, export it using:

```sh
poetry run python -m scripts.convert_to_onnx
```
- This script loads the PyTorch model, wraps it with preprocessing, and exports it to ONNX format.
- Make sure `weights/pytorch_model_weights.pth` exists before running this script.

---

## Local Model Inference

You can test the ONNX model directly using the provided script:

```sh
poetry run python -m src.model
```
- This will load a sample image and print the predicted class.

---

## Running the API Locally

Start the FastAPI server locally (requires the ONNX model):

```sh
make run
```
- The API will be available at `http://localhost:8000`.
- Health check: `GET /health`
- Prediction: `POST /v1/predict` (see below for payload)

---

## Testing the API Locally

Run the test suite (includes **local** model and API endpoint tests):

```sh
poetry run python -m tests.test
```

Or use the provided Makefile:

```sh
make test
```
- This will run ruff lint and format checks, as well as all tests.

---

## Docker Deployment

Build and run the API using Docker Compose:

```sh
make build
make run
```
- The API will be available at `http://localhost:8000` inside the container.
- To stop the container: simply press `Ctrl+C` in the terminal where it's running.

---

## Testing the Deployed API (Remote)

To test a deployed (e.g., Cerebrium) API endpoint, use `tests/test_server.py`:

```sh
poetry run python -m tests.test_server --api_key <YOUR_API_KEY>
```
- By default, this sends a sample image to the remote endpoint defined in the script.
- To test with a custom image:
  ```sh
  poetry run python -m tests.test_server --api_key <YOUR_API_KEY> --image_path path/to/image.jpg
  ```
- To run a set of preset tests:
```sh
poetry run python -m tests.test_server --api_key <YOUR_API_KEY> --run_tests
```

---

## Makefile Commands

- `make build` — Build the Docker image using Compose
- `make run` — Run the API in Docker
- `make test` — Run all tests (format, lint, and test)
- `make lint` — Lint code with ruff
- `make format` — Format code with ruff
- `make deploy` — Deploy to Cerebrium (requires credentials)

---

## Troubleshooting

- **Missing ONNX model:** Run `poetry run python -m scripts.convert_to_onnx` to generate `models/model.onnx`.
- **Missing weights:** Ensure `weights/pytorch_model_weights.pth` is present before exporting ONNX.
- **Docker build issues:** Make sure Docker is installed and running. Use `make build` for a clean build.
- **API key for remote tests:** Obtain your API key from the submitted assessment.
- **Poetry version:** Please make sure you are using Poetry version 1.8.3 and not a version greater than 2.0.

---

## API Usage Example

**POST /v1/predict**
- Request body (JSON):
  ```json
  {
    "image": "<base64-encoded-image>"
  }
  ```
- Response:
  ```json
  {
    "predicted_class": "mud turtle",
    "predicted_class_id": 35
  }
  ```

---

Thank you for your time!
