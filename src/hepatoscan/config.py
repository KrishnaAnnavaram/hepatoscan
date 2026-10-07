"""Settings from environment variables (and an optional local ``.env`` file)."""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

ENV_VARS = (
    "HEPATOSCAN_DATA_DIR",
    "HEPATOSCAN_RESULTS_DIR",
    "HEPATOSCAN_MODEL_PATH",
    "HEPATOSCAN_DEVICE",
    "HEPATOSCAN_API_TOKEN",
    "HEPATOSCAN_MAX_UPLOAD_MB",
    "HEPATOSCAN_AUDIT_LOG",
    "HEPATOSCAN_LLM_PROVIDER",
    "HEPATOSCAN_LLM_BASE_URL",
    "HEPATOSCAN_LLM_MODEL",
    "HEPATOSCAN_LLM_API_KEY",
    "HEPATOSCAN_LLM_TIMEOUT_S",
    "HEPATOSCAN_MAX_HISTORY",
    "HEPATOSCAN_SESSION_TTL_S",
)


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    data_dir: Path = Path("data/synthetic")
    results_dir: Path = Path("results")
    model_path: Path | None = None
    device: str = "auto"
    api_token: str | None = Field(default=None, repr=False)
    max_upload_mb: float = Field(64.0, gt=0)
    audit_log: Path = Path("results/audit.jsonl")
    llm_provider: str = "offline"  # "offline" or "openai" (any OpenAI-compatible server, Gemini included)
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str | None = Field(default=None, repr=False)
    llm_timeout_s: float = Field(30.0, gt=0)
    max_history: int = Field(6, ge=0)  # user/assistant exchanges kept per session
    session_ttl_s: float = Field(1800.0, gt=0)


def load_dotenv(path: str | os.PathLike = ".env") -> list[str]:
    """Read known ``KEY=VALUE`` lines from a local ``.env`` file. Variables already set win."""
    p = Path(path)
    if not p.is_file():
        return []
    loaded = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key in ENV_VARS and value and not os.environ.get(key):
            os.environ[key] = value.strip("\"'")
            loaded.append(key)
    return loaded


def settings_from_env() -> Settings:
    env = os.environ
    values: dict = {}
    mapping = {
        "HEPATOSCAN_DATA_DIR": "data_dir",
        "HEPATOSCAN_RESULTS_DIR": "results_dir",
        "HEPATOSCAN_MODEL_PATH": "model_path",
        "HEPATOSCAN_DEVICE": "device",
        "HEPATOSCAN_API_TOKEN": "api_token",
        "HEPATOSCAN_MAX_UPLOAD_MB": "max_upload_mb",
        "HEPATOSCAN_AUDIT_LOG": "audit_log",
        "HEPATOSCAN_LLM_PROVIDER": "llm_provider",
        "HEPATOSCAN_LLM_BASE_URL": "llm_base_url",
        "HEPATOSCAN_LLM_MODEL": "llm_model",
        "HEPATOSCAN_LLM_API_KEY": "llm_api_key",
        "HEPATOSCAN_LLM_TIMEOUT_S": "llm_timeout_s",
        "HEPATOSCAN_MAX_HISTORY": "max_history",
        "HEPATOSCAN_SESSION_TTL_S": "session_ttl_s",
    }
    for var, field in mapping.items():
        if env.get(var):
            values[field] = env[var]
    return Settings.model_validate(values)


def resolve_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
