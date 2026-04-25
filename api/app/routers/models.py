"""Models router — /v1/models endpoint.

Lists all models registered in the ModelRegistry.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.models.responses import ModelListResponse
from app.routers._registry import get_shared_registry

router = APIRouter(prefix="/v1")


@router.get("/models")
async def list_models():
    """Return a list of all available models."""
    registry = get_shared_registry()
    models = registry.list_models()
    body = ModelListResponse(data=models)
    return JSONResponse(status_code=200, content=body.model_dump())
