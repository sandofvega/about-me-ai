from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    chat_model: str = "gemini-2.5-flash-lite"
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = Field(..., env="QDRANT_API_KEY")
    qdrant_collection: str = "about_me"

    class Config:
        env_file = ".env"

settings = Settings()