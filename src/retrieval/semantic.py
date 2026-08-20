"""Semantic (vector) search over the vector store."""
from __future__ import annotations

from ..embeddings.embedder import Embedder
from ..ingestion.chunker import Chunk
from ..vectorstore.base import VectorStore


class SemanticSearch:
    def __init__(self, store: VectorStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def search(
        self,
        query: str,
        k: int,
        *,
        owner_id: str = "",
        tenant_id: str = "",
        trusted_only: bool = True,
    ) -> list[tuple[Chunk, float]]:
        if self.store.count() == 0:
            return []
        qvec = self.embedder.embed_query(query)
        return self.store.search(
            qvec,
            k,
            owner_id=owner_id,
            tenant_id=tenant_id,
            trusted_only=trusted_only,
        )