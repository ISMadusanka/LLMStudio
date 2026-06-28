"""Application settings.

Resolution order (highest priority first):
    1. Constructor / init kwargs
    2. Environment variables (prefix ``LLMSTUDIO_``, nested via ``__``)
    3. ``.env`` file
    4. ``config/default.yaml`` (+ optional ``$LLMSTUDIO_HOME/config.yaml`` override)
    5. Field defaults baked into the models below

Because every field has a sensible default here, the app boots even if the YAML
files are missing — the YAML simply documents and overrides those defaults.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from llmstudio.config.paths import ResolvedPaths, default_config_path, find_repo_root

# ---------------------------------------------------------------------------
# Section models (mirror config/default.yaml)
# ---------------------------------------------------------------------------


class AppConfig(BaseModel):
    name: str = "LLM Studio"
    environment: str = "local"


class PathsConfig(BaseModel):
    home: str = "workspace"
    uploads: str = "uploads"
    datasets: str = "datasets"
    runs: str = "runs"
    models: str = "models"
    hf_cache: str = "hf_cache"
    logs: str = "logs"
    registry_db: str = "registry.sqlite"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 7860
    share: bool = False
    auth: Optional[str] = None
    show_error: bool = True

    def auth_tuple(self) -> Optional[tuple[str, str]]:
        """Parse ``"user:password"`` into the tuple Gradio expects."""
        if not self.auth or ":" not in self.auth:
            return None
        user, _, password = self.auth.partition(":")
        return (user, password)


class AssistantConfig(BaseModel):
    enabled: bool = True
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    fallback_model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    min_vram_gb_for_primary: float = 18.0
    load_in_4bit: bool = True
    max_new_tokens: int = 1024
    temperature: float = 0.4
    top_p: float = 0.9
    keep_resident: bool = False


class TrainingDefaults(BaseModel):
    seed: int = 3407
    max_seq_length: int = 2048
    eval_ratio: float = 0.05
    logging_steps: int = 1
    save_steps: int = 50
    save_total_limit: int = 3
    default_export_format: str = "lora"
    packing: bool = False
    gradient_checkpointing: str = "unsloth"


class GpuPolicy(BaseModel):
    vram_safety_factor: float = 0.85
    qlora_threshold_gb: float = 24.0
    cpu_offload_allowed: bool = False


class DownloadPolicy(BaseModel):
    prefer_unsloth_4bit: bool = True
    allow_official_fallback: bool = True


# ---------------------------------------------------------------------------
# YAML settings source
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_yaml_config() -> dict[str, Any]:
    """Load ``default.yaml`` and deep-merge an optional user ``config.yaml``."""
    data: dict[str, Any] = {}
    default_path = default_config_path()
    if default_path.exists():
        with open(default_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

    home_env = os.environ.get("LLMSTUDIO_HOME")
    if home_env:
        user_cfg = Path(home_env).expanduser() / "config.yaml"
        if user_cfg.exists():
            with open(user_cfg, encoding="utf-8") as fh:
                data = _deep_merge(data, yaml.safe_load(fh) or {})
    return data


class _YamlSource(PydanticBaseSettingsSource):
    """Feeds parsed YAML into the pydantic-settings source chain."""

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        self._data = _load_yaml_config()

    def get_field_value(self, field, field_name: str):  # noqa: ANN001
        return self._data.get(field_name), field_name, False

    def prepare_field_value(self, field_name, field, value, value_is_complex):  # noqa: ANN001
        return value

    def __call__(self) -> dict[str, Any]:
        return {k: v for k, v in self._data.items() if v is not None}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LLMSTUDIO_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app: AppConfig = Field(default_factory=AppConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    assistant: AssistantConfig = Field(default_factory=AssistantConfig)
    training: TrainingDefaults = Field(default_factory=TrainingDefaults)
    gpu: GpuPolicy = Field(default_factory=GpuPolicy)
    download: DownloadPolicy = Field(default_factory=DownloadPolicy)
    log_level: str = "INFO"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,  # noqa: ANN001
        init_settings,  # noqa: ANN001
        env_settings,  # noqa: ANN001
        dotenv_settings,  # noqa: ANN001
        file_secret_settings,  # noqa: ANN001
    ):
        # init > env > .env > yaml > secrets. Sources are deep-merged by
        # pydantic-settings, so an env var like LLMSTUDIO_SERVER__PORT overrides
        # only that key, leaving the rest of `server` from YAML intact.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _YamlSource(settings_cls),
            file_secret_settings,
        )

    # -- derived paths ------------------------------------------------------
    @property
    def home_root(self) -> Path:
        env = os.environ.get("LLMSTUDIO_HOME")
        if env:
            return Path(env).expanduser().resolve()
        home = Path(self.paths.home).expanduser()
        if home.is_absolute():
            return home.resolve()
        return (find_repo_root() / home).resolve()

    @property
    def resolved_paths(self) -> ResolvedPaths:
        return ResolvedPaths.build(self.home_root, self.paths)

    def ensure_dirs(self) -> "Settings":
        self.resolved_paths.ensure_dirs()
        return self

    def hf_token(self) -> Optional[str]:
        """HF auth token from the usual env vars (for gated model downloads)."""
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


def reload_settings() -> Settings:
    """Clear the cache and rebuild settings (useful after editing .env/YAML)."""
    get_settings.cache_clear()
    return get_settings()
