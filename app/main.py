from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ai import ask

app = FastAPI()


class ChatMessageModel(BaseModel):
    role: str
    text: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessageModel] | None = None


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        reply = await ask(
            request.message,
            [msg.dict() for msg in request.history] if request.history else None,
        )
        return ChatResponse(reply=reply)

    except Exception:
        raise HTTPException(status_code=500, detail="Something went wrong.")
