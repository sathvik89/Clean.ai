"""Central configuration for Clean.ai.

Everything tunable lives here so the rest of the codebase never reads
environment variables directly.
"""

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str = ""
    groq_model_name: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.0

    # where run history, case memory and user accounts live
    data_dir: Path = PROJECT_ROOT / "storage"

    # agent loop guardrails
    max_tool_rounds: int = 20
    max_validation_rounds: int = 5
    max_step_attempts: int = 3
    max_stagnation_rounds: int = 2

    # safety: a single tool call may never delete more than this share of rows
    max_row_drop_pct: float = 40.0

    # context budget (approximate tokens) for the executor turn
    ledger_max_entries: int = 15
    tool_output_max_chars: int = 2400
    cards_max_per_prompt: int = 18

    @property
    def runs_db_path(self) -> Path:
        return self.data_dir / "runs.sqlite3"

    @property
    def users_db_path(self) -> Path:
        return self.data_dir / "users.sqlite3"

    @property
    def cases_db_path(self) -> Path:
        return self.data_dir / "cases.sqlite3"

    @property
    def checkpoints_db_path(self) -> Path:
        return self.data_dir / "checkpoints.sqlite3"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
