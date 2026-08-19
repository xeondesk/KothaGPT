from fastapi import APIRouter, HTTPException

from ...core import backend_factory
from ..schemas import EmbeddingRequest, EmbeddingResponse

router = APIRouter(prefix="/v1/embeddings", tags=["embeddings"])


@router.post("", response_model=EmbeddingResponse)
def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    backend = backend_factory.create()
    inputs = request.input if isinstance(request.input, list) else [request.input]
    if not inputs:
        raise HTTPException(status_code=422, detail="input must not be empty")
    return backend.embed(request.model, inputs)