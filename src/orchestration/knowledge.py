"""Knowledge base bootstrap + on-demand source ingestion.

Bundled demo evidence is ingested on first use so ClinKey works offline;
real selected guidelines are downloaded/cached/ingested on demand.
"""
from __future__ import annotations

import json

from ..config.settings import DATA_DIR, get_settings
from ..embeddings.embedder import Embedder
from ..ingestion.pipeline import IngestionPipeline
from ..sources.models import Source
from ..vectorstore import get_vectorstore


class KnowledgeBase:
    def __init__(self):
        self.settings = get_settings()
        self.store = get_vectorstore()
        self.embedder = Embedder()
        self.pipeline = IngestionPipeline(self.store, self.embedder)
        self._bootstrapped = False

    @property
    def chunk_count(self) -> int:
        return self.store.count()

    def source_ids_ingested(self) -> set[str]:
        return {c.source_id for c in self.store.all_chunks()}

    def ensure_ready(self) -> None:
        """Ingest bundled demo knowledge if nothing is present yet."""
        if self._bootstrapped:
            return
        self._bootstrapped = True
        if self.store.count() > 0:
            return
        path = DATA_DIR / "demo_knowledge" / "guidelines.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            for doc in data.get("documents", []):
                self.pipeline.ingest_bundled_document(doc)

    def ensure_sources(self, sources: list[Source]) -> list[dict]:
        """Acquire + ingest the given validated sources. Returns status list."""
        self.ensure_ready()
        statuses = []
        for src in sources:
            if src.source_id in self.source_ids_ingested():
                statuses.append({"source_id": src.source_id, "status": "already_ingested", "chunks_added": 0})
                continue
            statuses.append(self.pipeline.ingest_source(src))
        return statuses