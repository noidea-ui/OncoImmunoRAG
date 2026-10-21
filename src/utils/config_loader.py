"""Load and validate the project's YAML configuration.

Usage:
    from utils.config_loader import load_config

    cfg = load_config()                 # auto-resolves <project_root>/config.yaml
    cfg = load_config("path/to/cfg.yaml")

The loader returns an :class:`AppConfig` dataclass with strongly-typed fields,
so downstream code can rely on ``cfg.chunk_size`` being an ``int``.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Default config file name, looked up at the project root (parent of ``src/``).
_DEFAULT_CONFIG_NAME = "config.yaml"

# Environment variable that, when set, overrides ``openai_api_key`` from the YAML.
_OPENAI_API_KEY_ENV = "OPENAI_API_KEY"


def _project_root() -> Path:
    """Return the project root by walking up from this file.

    Layout assumed: ``<root>/src/utils/config_loader.py``.
    """
    return Path(__file__).resolve().parents[2]


def resolve_config_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the config path. Defaults to ``<project_root>/config.yaml``."""
    if path is None:
        return _project_root() / _DEFAULT_CONFIG_NAME
    return Path(path).expanduser().resolve()


@dataclass(frozen=True)
class AppConfig:
    """Strongly-typed view of ``config.yaml``.

    ``openai_api_key`` may be empty; call :meth:`effective_openai_api_key` to
    fall back to the ``OPENAI_API_KEY`` environment variable.
    """

    openai_api_key: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    # Preserve any extra YAML keys the user adds — handy for forward-compat.
    extras: dict[str, Any] = field(default_factory=dict)

    def effective_openai_api_key(self) -> str:
        """Return the OpenAI key, preferring the ``OPENAI_API_KEY`` env var."""
        env_key = os.environ.get(_OPENAI_API_KEY_ENV, "").strip()
        return env_key or self.openai_api_key.strip()

    # --- light-weight accessors for downstream modules ---------------------
    @property
    def has_openai_api_key(self) -> bool:
        return bool(self.effective_openai_api_key())


def _coerce(raw: dict[str, Any]) -> AppConfig:
    """Validate raw YAML mapping and produce an :class:`AppConfig`."""
    if not isinstance(raw, dict):
        raise ValueError(f"Top-level YAML must be a mapping, got {type(raw).__name__}")

    missing = [k for k in ("embedding_model", "chunk_size", "chunk_overlap") if k not in raw]
    if missing:
        raise ValueError(f"config.yaml is missing required keys: {missing}")

    try:
        chunk_size = int(raw["chunk_size"])
        chunk_overlap = int(raw["chunk_overlap"])
    except (TypeError, ValueError) as exc:
        raise ValueError("chunk_size and chunk_overlap must be integers") from exc

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
        )

    embedding_model = str(raw["embedding_model"]).strip()
    if not embedding_model:
        raise ValueError("embedding_model must not be empty")

    openai_api_key = str(raw.get("openai_api_key", "")).strip()
    if not openai_api_key and not os.environ.get(_OPENAI_API_KEY_ENV):
        warnings.warn(
            "openai_api_key is empty in config.yaml and OPENAI_API_KEY is not set; "
            "OpenAI-backed calls will fail until one is provided.",
            stacklevel=2,
        )

    known = {"openai_api_key", "embedding_model", "chunk_size", "chunk_overlap"}
    extras = {k: v for k, v in raw.items() if k not in known}

    return AppConfig(
        openai_api_key=openai_api_key,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        extras=extras,
    )


def load_config(
    path: str | os.PathLike[str] | None = None,
    *,
    raise_if_missing: bool = True,
) -> AppConfig:
    """Load, validate, and return the project configuration.

    Args:
        path: Optional explicit path. Defaults to ``<project_root>/config.yaml``.
        raise_if_missing: When ``False``, return a default-constructed config
            if the file does not exist (useful for tests).
    """
    cfg_path = resolve_config_path(path)
    if not cfg_path.is_file():
        if raise_if_missing:
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        return _coerce({})

    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return _coerce(raw)


if __name__ == "__main__":
    # Quick CLI: `poetry run python -m utils.config_loader`
    cfg = load_config()
    print(
        f"embedding_model={cfg.embedding_model} | "
        f"chunk_size={cfg.chunk_size} | chunk_overlap={cfg.chunk_overlap} | "
        f"openai_key_present={cfg.has_openai_api_key}"
    )
