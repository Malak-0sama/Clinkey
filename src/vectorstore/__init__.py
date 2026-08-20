"""Vector store package with a provider factory."""
from __future__ import annotations

from ..config.settings import get_settings
from .base import VectorStore
from .memory import MemoryStore


def get_vectorstore() -> VectorStore:
    settings = get_settings()
    provider = settings.vectorstore_effective_provider
    if provider == "chroma":
        try:
            from .chroma import ChromaStore

            return ChromaStore(settings)
        except Exception:  # noqa: BLE001
            pass
    return MemoryStore()


__all__ = ["VectorStore", "MemoryStore", "get_vectorstore"]