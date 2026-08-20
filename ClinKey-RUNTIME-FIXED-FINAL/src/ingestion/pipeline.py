"""Ingestion pipeline: acquisition -> cleaning -> chunking -> embedding -> store."""
from __future__ import annotations

from ..config.settings import get_settings
from ..embeddings.embedder import Embedder
from ..sources.models import Source
from ..vectorstore.base import VectorStore
from .cache import SourceCache
from .chunker import Chunk, SectionAwareChunker
from .cleaner import TextCleaner
from .downloader import DownloadError, Downloader
from .quality import ChunkQualityFilter


class IngestionPipeline:
    def __init__(self, store: VectorStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder
        self.chunker = SectionAwareChunker()
        self.cleaner = TextCleaner()
        self.cache = SourceCache()
        self.downloader = Downloader()
        self.quality = ChunkQualityFilter()
        self.last_quality_metrics: dict = {}

    def ingest_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        accepted, metrics = self.quality.filter_chunks(chunks)
        self.last_quality_metrics = metrics
        if not accepted:
            return 0
        existing = {c.chunk_id for c in self.store.all_chunks()}
        new_chunks = [c for c in accepted if c.chunk_id not in existing]
        if not new_chunks:
            return 0
        vecs = self.embedder.embed([c.text for c in new_chunks])
        self.store.add(new_chunks, vecs)
        return len(new_chunks)

    def ingest_source(self, source: Source) -> dict:
        """Acquire (download or cache) + ingest a validated source.

        Returns a status dict: {source_id, status, chunks_added}.
        """
        status = {"source_id": source.source_id, "status": "ok", "chunks_added": 0}

        try:
            if self.cache.is_cached(source.source_id):
                text = self.cache.get(source.source_id)
                status["acquired"] = "cached"
            else:
                text = self.downloader.fetch(source)
                text = self.cleaner.clean(text)
                if not text.strip():
                    raise DownloadError("cleaned text is empty")
                self.cache.put(source, text)
                status["acquired"] = "downloaded"

            chunks = self.chunker.chunk(source, text)
            status["chunks_added"] = self.ingest_chunks(chunks)
        except DownloadError as exc:
            status["status"] = "download_failed"
            status["error"] = str(exc)[:200]
        except Exception as exc:  # noqa: BLE001
            status["status"] = "error"
            status["error"] = str(exc)[:200]

        return status

    def ingest_bundled_document(self, doc: dict) -> int:
        """Ingest a pre-chunked bundled document (data/demo_knowledge)."""
        chunks: list[Chunk] = []
        for c in doc.get("chunks", []):
            chunks.append(
                Chunk(
                    chunk_id=c["chunk_id"],
                    source_id=doc["source_id"],
                    document_name=doc["title"],
                    organization=doc["organization"],
                    page_number=int(c.get("page", 1)),
                    section_title=c.get("section", ""),
                    source_url=doc.get("official_url", ""),
                    publication_date=doc.get("publication_date", ""),
                    version=doc.get("version", ""),
                    text=c["text"],
                )
            )
        return self.ingest_chunks(chunks)