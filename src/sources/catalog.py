"""Approved Source Catalog — an allowlist of official medical guidelines."""
from __future__ import annotations

import json

from ..config.settings import CONFIG_DIR
from .models import Source


class SourceCatalog:
    def __init__(self, path=None):
        self.path = path or (CONFIG_DIR / "approved_sources.json")
        self._sources: dict[str, Source] = {}
        self._load()

    def _load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        for item in data.get("sources", []):
            src = Source.from_dict(item)
            self._sources[src.source_id] = src

    def all(self) -> list[Source]:
        return list(self._sources.values())

    def allowed(self) -> list[Source]:
        return [s for s in self._sources.values() if s.allowed]

    def get(self, source_id: str) -> Source | None:
        return self._sources.get(source_id)

    def ids(self) -> set[str]:
        return set(self._sources.keys())