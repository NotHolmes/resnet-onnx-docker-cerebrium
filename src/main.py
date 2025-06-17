import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.model import OnnxModel
from src.routers import health_check
from src.routers.v1 import predict

handler = logging.StreamHandler()
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)

ALLOW_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[Any, Any]:
    """Manages the startup and shutdown events for the FastAPI application.

    Args:
        app (FastAPI): The FastAPI application instance.
    """
    model_path = Path(__file__).parent.parent / "models" / "model.onnx"
    model_path = str(model_path.resolve())
    app.state.model = OnnxModel(model_path, True)
    yield


app = FastAPI(title="MTailor API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=ALLOW_METHODS,
    allow_headers=["*"],
)
app.include_router(health_check.router)
app.include_router(predict.router)
