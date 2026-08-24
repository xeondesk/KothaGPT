from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ...core import backend_factory
from ..schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    LegacyChatRequest,
    LegacyChatResponse,
)

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
) -> StreamingResponse | ChatCompletionResponse:
    backend = backend_factory.create()
    if request.stream:
        return StreamingResponse(
            _sse_stream(backend.chat_stream(request)),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    try:
        return backend.chat(request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/chat/stream")
async def chat_stream(request: ChatCompletionRequest) -> StreamingResponse:
    """Convenience endpoint: always streams, regardless of the stream flag."""
    backend = backend_factory.create()
    return StreamingResponse(
        _sse_stream(backend.chat_stream(request)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/chat", response_model=LegacyChatResponse)
async def chat(request: LegacyChatRequest) -> LegacyChatResponse:
    """Legacy single-message chat, kept for backward compatibility."""
    backend = backend_factory.create()
    chat_req = ChatCompletionRequest(
        model=request.model,
        messages=[{"role": "user", "content": request.message}],
    )
    response = backend.chat(chat_req)
    return LegacyChatResponse(
        model=request.model,
        message=request.message,
        output=response.choices[0].message.content,
    )


def _sse_stream(chunks) -> StreamingResponse:  # type: ignore[no-untyped-def]
    def generate():
        for chunk in chunks:
            yield f"data: {json.dumps(chunk.model_dump(), ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return generate()
