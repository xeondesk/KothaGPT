from fastapi import APIRouter

from ...core import backend_factory
from ..schemas import RerankRequest, RerankResponse

router = APIRouter(prefix="/v1/rerank", tags=["rerank"])


@router.post("", response_model=RerankResponse)
def rerank(request: RerankRequest) -> RerankResponse:
    backend = backend_factory.create()
    return backend.rerank(request.model, request.query, request.documents, request.top_n)