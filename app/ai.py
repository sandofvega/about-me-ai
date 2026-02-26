from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pathlib import Path
from core.config import settings

# Load personal data
personal_data = Path("app/data/me.md").read_text(encoding="utf-8")

MAX_HISTORY_PAIRS = 5

system_prompt = f"""
You are a personal assistant that answers questions ONLY about Yasin.

RULES:
- Only use the information provided below to answer questions
- If a question is not about Yasin or cannot be answered from the provided data, politely decline and explain that you can only answer questions about Yasin
- Keep responses concise and factual
- Do not make up or guess any information

PERSONAL DATA:
{personal_data}
"""


class ChatMessage:
    def __init__(self, role: str, text: str):
        self.role = role
        self.text = text


async def ask(user_message: str, history: list[dict] | None = None) -> str:
    model = ChatGoogleGenerativeAI(
        model=settings.chat_model,
        api_key=settings.gemini_api_key
    )

    history = history or []
    trimmed_history = history[-(MAX_HISTORY_PAIRS * 2):]

    history_messages = []
    for msg in trimmed_history:
        if msg["role"] == "human":
            history_messages.append(HumanMessage(content=msg["text"]))
        else:
            history_messages.append(AIMessage(content=msg["text"]))

    response = await model.ainvoke(
        [
            SystemMessage(content=system_prompt),
            *history_messages,
            HumanMessage(content=user_message)
        ]
    )

    return response.content
