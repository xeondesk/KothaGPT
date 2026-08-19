from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Kotha GPT API", version="0.1.0")

class ChatRequest(BaseModel):
    message: str
    model: str = "kothagpt"

@app.get("/health")
def health():
    return {"status": "ok", "service": "kothagpt-api"}

@app.post("/v1/chat")
def chat(request: ChatRequest):
    # Replace this stub with the inference gateway.
    return {
        "model": request.model,
        "message": request.message,
        "output": "Inference backend not configured yet."
    }
