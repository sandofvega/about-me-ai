from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pathlib import Path
from core.config import settings
from lib.qdrant import search_similar

MAX_HISTORY_PAIRS = 5


async def ask(user_message: str, history: list[dict] | None = None) -> str:
    model = ChatGoogleGenerativeAI(
        model=settings.chat_model,
        api_key=settings.gemini_api_key
    )

    # History building
    history = history or []
    trimmed_history = history[-(MAX_HISTORY_PAIRS * 2):]

    history_messages = []
    for msg in trimmed_history:
        if msg["role"] == "human":
            history_messages.append(HumanMessage(content=msg["text"]))
        else:
            history_messages.append(AIMessage(content=msg["text"]))
    # History building END

    similar_data = search_similar(user_message)

    system_prompt = f"""
You are a factual, constraint-bound assistant specialized in answering questions strictly about the individual named Yasin.

OBJECTIVE:
Provide accurate, concise answers using ONLY the supplied PERSONAL DATA.

CONTEXT:
{similar_data}

RULES:
1. Source Restriction:
    - Use exclusively the information in PERSONAL DATA.
    - Do NOT infer, assume, or add external knowledge.

2. Scope Control:
    - Only answer questions directly related to Yasin.
    - If the question is unrelated or outside the data, respond with:
        "I can only answer questions about Yasin based on the provided information."

3. Accuracy:
    - If the answer is not explicitly supported by the data, say:
        "That information is not available in the provided data."

4. Response Style:
    - Always respond in a full sentence.
    - Use simple, natural phrasing.
    - Keep it concise but complete.

5. Consistency Check:
    - Before answering, verify the answer is explicitly grounded in the data.

OUTPUT FORMAT:
    - Direct answer only.
    - No explanations unless necessary for clarity.
    """

    response = await model.ainvoke(
        [
            SystemMessage(content=system_prompt),
            *history_messages,
            HumanMessage(content=user_message)
        ]
    )

    return response.content
