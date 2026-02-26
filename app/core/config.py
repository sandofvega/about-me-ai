from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    chat_model: str = "gemini-2.5-flash-lite"
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")

    class Config:
        env_file = ".env"

settings = Settings()