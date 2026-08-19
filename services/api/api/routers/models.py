from fastapi import APIRouter

from ...core import backend_factory
from ..schemas import ModelList

router = APIRouter(prefix="/v1/models", tags=["models"])


@router.get("", response_model=ModelList)
def list_models() -> ModelList:
    backend = backend_factory.create()
    return ModelList(data=backend.list_models())