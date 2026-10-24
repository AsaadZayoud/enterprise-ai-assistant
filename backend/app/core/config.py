from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Centralized Application Configuration.
    Validates enviromment variables as startup
    """

    # Enviroment & Server
    PROJECT_NAME: str = "Enterprise AI Assistent"
    DEBUG: bool = False

    # LLM & API Keys
    GEMINI_API_KEY: str

    # Vector DB 
    CHROMA_DB_PATH: str = "backend/chroma_db"

    # Configuration to read from .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encondig="utf-8",
        extra="ignore"  # Ignores extra variables in .env if not declared above
    )


settings = Settings()