import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from core.config import settings
from lib.ai import ask

logger = logging.getLogger(__name__)

app = FastAPI()

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


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

    except Exception as e:
        logger.exception(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail="Something went wrong.")


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }
