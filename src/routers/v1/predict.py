import base64
import logging
from io import BytesIO

from fastapi import APIRouter, Request
from PIL import Image
from pydantic import BaseModel

from src.class_labels import CLASS_LABELS
from src.model import OnnxModel

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/v1",
    responses={200: {"model": {}}},
)


class PredictRequest(BaseModel):
    """Request model for prediction endpoint."""

    image: str  # base64-encoded image

    class Config:
        """Pydantic configuration."""

        schema_extra = {"example": {"image": "iVBORw0KGgoAAAANSUhEUgAA..."}}


class PredictResponse(BaseModel):
    """Response model for prediction endpoint."""

    predicted_class: str
    predicted_class_id: int

    class Config:
        """Pydantic configuration."""

        schema_extra = {
            "example": {
                "predicted_class": "mud turtle",
                "predicted_class_id": 35,
            }
        }


@router.post("/predict", response_model=PredictResponse)
async def predict(input_data: PredictRequest, request: Request) -> PredictResponse:
    """Predict the class of an image.

    Args:
        input_data (PredictRequest): The request data containing the base64-encoded image.
        request (Request): The FastAPI request object, used to access app state.

    Returns:
        PredictResponse: The predicted class and class ID.
    """
    # Decode the base64 image
    try:
        image_data = base64.b64decode(input_data.image)
    except (base64.binascii.Error, ValueError) as err:
        logger.exception("Failed to decode base64 image")
        error_message = "Invalid base64 image"
        raise ValueError(error_message) from err

    try:
        image = Image.open(BytesIO(image_data))
    except (OSError, Image.UnidentifiedImageError) as err:
        logger.exception("Failed to process image")
        error_message = "Invalid image data"
        raise ValueError(error_message) from err

    model: OnnxModel = request.app.state.model

    predicted_class_id = model.predict_index(image)
    predicted_class = CLASS_LABELS[predicted_class_id]

    return PredictResponse(predicted_class=predicted_class, predicted_class_id=predicted_class_id)
