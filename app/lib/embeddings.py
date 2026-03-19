from langchain_google_genai import GoogleGenerativeAIEmbeddings
from core.config import settings

_embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=settings.gemini_api_key
)

def get_embedding(text: str) -> list[float]:
    """
    Convert text into embedding vector.
    """
    
    normalized = text.strip()
    return _embeddings.embed_query(normalized)
