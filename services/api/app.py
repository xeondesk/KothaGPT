from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Own AI API", version="0.1.0")

class ChatRequest(BaseModel):
    message: str
    model: str = "own-ai"

@app.get("/health")
def health():
    return {"status": "ok", "service": "own-ai-api"}

@app.post("/v1/chat")
def chat(request: ChatRequest):
    # Replace this stub with the inference gateway.
    return {
        "model": request.model,
        "message": request.message,
        "output": "Inference backend not configured yet."
    }
