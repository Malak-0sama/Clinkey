"""Centralized configuration.

Loads configs/settings.yaml and overlays secrets from the environment.
Secrets (API keys) are NEVER hard-coded and never written to logs.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]          # ClinKey/
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
ASSETS_DIR = ROOT / "assets"


def _load_env() -> dict[str, str]:
    """Load .env if present (simple loader; no external dependency)."""
    env = {}
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            env[key.strip()] = val.strip().strip('"').strip("'")
    return env


class Settings:
    """Flat attribute access over the merged YAML + env configuration."""

    def __init__(self, data: dict[str, Any]):
        self._raw = data
        self._flat: dict[str, Any] = {}
        self._flatten(data, "", self._flat)
        for k, v in self._flat.items():
            setattr(self, k, v)

    @staticmethod
    def _flatten(d: dict, prefix: str, out: dict) -> None:
        for k, v in d.items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}_{k}"
            if isinstance(v, dict):
                Settings._flatten(v, key, out)
            else:
                out[key] = v

    def get(self, dotted: str, default: Any = None) -> Any:
        return self._flat.get(dotted, default)

    # ---- paths ----
    @property
    def root(self) -> Path:
        return ROOT

    @property
    def cache_path(self) -> Path:
        return ROOT / self.sources_cache_dir

    @property
    def vector_db_path(self) -> Path:
        return ROOT / self.vectorstore_path

    @property
    def history_dir(self) -> Path:
        return ROOT / self.history_path

    @property
    def supported_language_codes(self) -> list[str]:
        return [lang["code"] for lang in self.language_supported]

    # ---- env-derived secrets / feature flags ----
    @property
    def gemini_api_key(self) -> str:
        return os.environ.get("GEMINI_API_KEY", "")

    @property
    def llm_effective_provider(self) -> str:
        """Resolve 'auto' -> 'gemini' if a key exists, else 'demo'."""
        provider = self.get("llm_provider", "auto")
        if provider == "auto":
            return "gemini" if self.gemini_api_key else "demo"
        return provider

    @property
    def embeddings_effective_provider(self) -> str:
        provider = self.get("embeddings_provider", "auto")
        if provider != "auto":
            return provider
        try:
            import sentence_transformers  # noqa: F401
            return "sentence-transformers"
        except Exception:
            return "hash"

    @property
    def vectorstore_effective_provider(self) -> str:
        provider = self.get("vectorstore_provider", "auto")
        if provider != "auto":
            return provider
        try:
            import chromadb  # noqa: F401
            return "chroma"
        except Exception:
            return "memory"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        cfg_file = CONFIG_DIR / "settings.yaml"
        data = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
        _settings = Settings(data)
        env = _load_env()
        for k, v in env.items():
            os.environ.setdefault(k, v)
        _validate_settings(_settings)
    return _settings


def _validate_settings(s: Settings) -> None:
    for key in (
        "confidence_thresholds_high",
        "confidence_thresholds_medium",
        "confidence_thresholds_low",
        "retrieval_heuristic_min_rerank",
        "retrieval_min_support_score",
    ):
        v = s.get(key)
        if v is None:
            continue
        if not (0.0 <= float(v) <= 1.0):
            raise ValueError(f"config {key}={v} must be in [0, 1]")