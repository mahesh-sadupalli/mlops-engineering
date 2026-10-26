"""Runtime configuration, read once from environment variables."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model: str = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
    effort: str = os.getenv("CLAUDE_EFFORT", "medium")
    chroma_dir: Path = Path(os.getenv("CHROMA_DIR", "chroma"))
    collection: str = os.getenv("CHROMA_COLLECTION", "engineering-docs")
    docs_dirs: list[Path] = field(
        default_factory=lambda: [
            Path(p) for p in os.getenv("DOCS_DIRS", "../..").split(":")
        ]
    )
    house_prices_url: str = os.getenv("HOUSE_PRICES_URL", "http://localhost:8000")
    cors_origins: list[str] = field(
        default_factory=lambda: os.getenv(
            "CORS_ORIGINS", "http://localhost:3000"
        ).split(",")
    )
    chunk_size: int = 1200
    top_k: int = 5
    max_turns: int = 8


settings = Settings()
