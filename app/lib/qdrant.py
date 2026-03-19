from core.config import settings
from qdrant_client import QdrantClient
from lib.embeddings import get_embedding

_client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key
)

def search_similar(text: str, limit: int = 5):
    """
    Get top N similar points from Qdrant based on input text.
    """

    vector = get_embedding(text)

    results = _client.query_points(
        collection_name=settings.qdrant_collection,
        query=vector,
        limit=limit,
    )

    return [point.payload.get("text") for point in results.points]
