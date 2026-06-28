"""Configuration package: settings + path resolution."""

from llmstudio.config.paths import (
    ResolvedPaths,
    default_config_path,
    find_config_dir,
    find_repo_root,
    models_catalog_path,
)
from llmstudio.config.settings import (
    AppConfig,
    AssistantConfig,
    DownloadPolicy,
    GpuPolicy,
    PathsConfig,
    ServerConfig,
    Settings,
    TrainingDefaults,
    get_settings,
    reload_settings,
)

__all__ = [
    "AppConfig",
    "AssistantConfig",
    "DownloadPolicy",
    "GpuPolicy",
    "PathsConfig",
    "ResolvedPaths",
    "ServerConfig",
    "Settings",
    "TrainingDefaults",
    "default_config_path",
    "find_config_dir",
    "find_repo_root",
    "get_settings",
    "models_catalog_path",
    "reload_settings",
]
