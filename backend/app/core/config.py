from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    app_name: str = "WeightAtlas"
    version: str = "0.1.0"
    host: str = "127.0.0.1"
    port: int = 8000
    frontend_dist: Path = Path(__file__).parent.parent.parent.parent / "frontend" / "dist"

    class Config:
        env_prefix = "WEIGHTATLAS_"


settings = Settings()
