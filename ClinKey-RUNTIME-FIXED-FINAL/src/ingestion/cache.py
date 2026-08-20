"""Source cache — never re-download / re-ingest the same guideline per query."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from ..config.settings import get_settings
from ..sources.models import Source


class SourceCache:
    def __init__(self):
        self.settings = get_settings()
        self.cache_dir: Path = self.settings.cache_path
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.cache_dir / "index.json"
        self._index = self._load_index()

    # ---- index ----
    def _load_index(self) -> dict:
        if self.index_path.exists():
            try:
                return json.loads(self.index_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_index(self) -> None:
        self.index_path.write_text(json.dumps(self._index, indent=2), encoding="utf-8")

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # ---- API ----
    def is_cached(self, source_id: str) -> bool:
        entry = self._index.get(source_id)
        if not entry:
            return False
        path = self.cache_dir / entry["filename"]
        return path.exists()

    def get(self, source_id: str) -> str | None:
        if not self.is_cached(source_id):
            return None
        entry = self._index[source_id]
        path = self.cache_dir / entry["filename"]
        return path.read_text(encoding="utf-8")

    def put(self, source: Source, text: str) -> str:
        digest = self._hash(text)
        filename = f"{source.source_id}.txt"
        (self.cache_dir / filename).write_text(text, encoding="utf-8")
        self._index[source.source_id] = {
            "source_id": source.source_id,
            "organization": source.organization,
            "title": source.title,
            "url": source.official_url,
            "version": source.version,
            "publication_date": source.publication_date,
            "document_hash": digest,
            "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._save_index()
        return digest

    def status(self) -> list[dict]:
        return list(self._index.values())