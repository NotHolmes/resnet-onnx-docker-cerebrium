import base64
from pathlib import Path

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from PIL import Image

from src.class_labels import CLASS_LABELS
from src.main import app
from src.model import OnnxModel

HTTP_OK = status.HTTP_200_OK
HTTP_BAD_REQUEST = status.HTTP_400_BAD_REQUEST
HTTP_UNPROCESSABLE_ENTITY = status.HTTP_422_UNPROCESSABLE_ENTITY

client = TestClient(app)


def test_health_check() -> None:
    """Test function to verify the behavior of the 'health_check' API endpoint.

    Args:
        mocker (MockerFixture): The pytest mocker fixture.
        fastapi_client (TestClient): FastAPI test client instance.

    """
    result = client.get("/health")
    assert result.status_code == status.HTTP_200_OK


def test_onnx_model_local() -> None:
    """Test local ONNX model inference with a sample image."""
    model_path = str(Path(__file__).parent.parent / "models" / "model.onnx")
    img_path = str(Path(__file__).parent.parent / "images" / "n01667114_mud_turtle.JPEG")

    model = OnnxModel(model_path)
    img = Image.open(img_path)
    pred_idx = model.predict_index(img)

    assert isinstance(pred_idx, int), "Prediction index should be int"
    assert 0 <= pred_idx < len(CLASS_LABELS), "Prediction index out of range"

    print(f"Local model prediction: {CLASS_LABELS[pred_idx]} (index {pred_idx})")


def test_api_predict() -> None:
    """Test FastAPI /v1/predict endpoint with a sample image using TestClient."""
    img_path = Path(__file__).parent.parent / "images" / "n01667114_mud_turtle.JPEG"

    with img_path.open("rb") as img_file:
        b64_image = base64.b64encode(img_file.read()).decode("utf-8")
    payload = {"image": b64_image}

    with client:
        response = client.post("/v1/predict", json=payload)
        assert response.status_code == HTTP_OK, f"API returned status {response.status_code}: {response.text}"

        data = response.json()
        assert "predicted_class" in data, "API response missing 'predicted_class' key"
        assert "predicted_class_id" in data, "API response missing 'predicted_class_id' key"

        print(f"API prediction: {data['predicted_class']} (index {data['predicted_class_id']})")

    print("API prediction test passed.")


def test_invalid_image() -> None:
    """Test API with invalid image data using TestClient."""
    payload = {"image": "not_base64_data"}

    with pytest.raises(ValueError, match="Invalid base64 image"):
        client.post("/v1/predict", json=payload)

    print("API correctly handled invalid image input.")


if __name__ == "__main__":
    print("Testing local ONNX model inference...")
    test_onnx_model_local()
    print("Testing API /v1/predict endpoint...")
    test_api_predict()
    print("Testing API with invalid image...")
    test_invalid_image()
    print("Testing health check endpoint...")
    test_health_check()
    print("All tests passed.")
