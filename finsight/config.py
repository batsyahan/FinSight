"""Runtime configuration, read from environment variables (and a .env file if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass

try:  # optional
    from dotenv import load_dotenv

    load_dotenv(override=True)  # the project's .env wins over stray global env vars
except Exception:  # pragma: no cover
    pass


@dataclass(frozen=True)
class Settings:
    llm: str = "mock"
    model: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    data: str = "sample"
    max_revisions: int = 2

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            llm=os.getenv("FINSIGHT_LLM", "mock").strip().lower(),
            model=os.getenv("FINSIGHT_MODEL", "").strip(),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            data=os.getenv("FINSIGHT_DATA", "sample").strip().lower(),
            max_revisions=int(os.getenv("FINSIGHT_MAX_REVISIONS", "2")),
        )
