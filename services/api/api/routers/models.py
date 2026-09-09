from fastapi import APIRouter

from ...core import backend_factory
from ..schemas import ModelList

try:
    from ml.inference.registry import get_registry
except ImportError:
    get_registry = None  # type: ignore

router = APIRouter(prefix="/v1/models", tags=["models"])


@router.get("", response_model=ModelList)
def list_models() -> ModelList:
    # Prefer registry when available (WS-9), fallback to backend mock for CI
    if get_registry is not None:
        try:
            from ..schemas import Model as APIModel

            regs = get_registry().list_models()
            if regs:
                return ModelList(data=[APIModel(id=r.id, description=r.name, context_window=r.context_window) for r in regs])
        except Exception:
            pass
    backend = backend_factory.create()
    return ModelList(data=backend.list_models())
