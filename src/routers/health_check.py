from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

router = APIRouter(
    responses={
        200: {
            "description": "Successful health check",
            "content": {"application/json": {"example": {"message": "OK"}}},
        }
    },
)


@router.get("/health")
async def healthcheck() -> JSONResponse:
    """Health check endpoint.

    Returns:
        JSONResponse: 200 status
    """
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "OK"},
    )
