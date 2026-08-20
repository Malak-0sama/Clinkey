"""In-process numpy vector store (offline fallback)."""
from __future__ import annotations

import numpy as np

from ..ingestion.chunker import Chunk
from .base import VectorStore
from .scores import cosine_to_unit_similarity


class MemoryStore(VectorStore):
    def __init__(self):
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        # de-duplicate by chunk_id
        existing = {c.chunk_id for c in self._chunks}
        new_chunks = [c for c in chunks if c.chunk_id not in existing]
        if not new_chunks:
            return
        idx = [i for i, c in enumerate(chunks) if c.chunk_id in {c2.chunk_id for c2 in new_chunks}]
        new_vecs = embeddings[idx]
        if self._vectors is None:
            self._vectors = new_vecs
        else:
            self._vectors = np.vstack([self._vectors, new_vecs])
        self._chunks.extend(new_chunks)

    def _authorized(self, chunk: Chunk, owner_id: str, tenant_id: str, trusted_only: bool) -> bool:
        if trusted_only and not chunk.trusted:
            return False
        # Official catalog chunks have empty owner; private chunks must match identity.
        if chunk.owner_id:
            if not owner_id or chunk.owner_id != owner_id:
                return False
        if chunk.tenant_id:
            if not tenant_id or chunk.tenant_id != tenant_id:
                return False
        return True

    def search(
        self,
        query_embedding: np.ndarray,
        k: int,
        *,
        owner_id: str = "",
        tenant_id: str = "",
        trusted_only: bool = True,
    ) -> list[tuple[Chunk, float]]:
        if self._vectors is None or len(self._chunks) == 0:
            return []
        q = query_embedding.astype("float32")
        sims = self._vectors @ q  # L2-normalized cosine in [-1, 1]
        order = np.argsort(-sims)
        results: list[tuple[Chunk, float]] = []
        for i in order:
            chunk = self._chunks[i]
            if not self._authorized(chunk, owner_id, tenant_id, trusted_only):
                continue
            results.append((chunk, cosine_to_unit_similarity(float(sims[i]))))
            if len(results) >= k:
                break
        return results

    def all_chunks(
        self,
        *,
        owner_id: str = "",
        tenant_id: str = "",
        trusted_only: bool = False,
    ) -> list[Chunk]:
        return [
            c for c in self._chunks
            if self._authorized(c, owner_id, tenant_id, trusted_only)
        ]

    def count(self) -> int:
        return len(self._chunks)

    def delete_source(self, source_id: str) -> None:
        keep_idx = [i for i, c in enumerate(self._chunks) if c.source_id != source_id]
        self._chunks = [self._chunks[i] for i in keep_idx]
        if self._vectors is not None and keep_idx:
            self._vectors = self._vectors[keep_idx]
        elif self._vectors is not None:
            self._vectors = None

    def health_check(self) -> bool:
        return self._vectors is not None and self._vectors.shape[0] == len(self._chunks)