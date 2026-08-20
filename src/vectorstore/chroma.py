"""ChromaDB-backed persistent vector store.

Chroma is configured with hnsw:space=cosine. Query distances are cosine
*distance* (0 = identical, 2 = opposite). We convert to cosine similarity:

    relevance = 1.0 - distance

S = 1.0 - max(0.0, min(1.0, d))   # contract: 1 identical, 0 orthogonal/opposite
"""
from __future__ import annotations

import numpy as np

from ..ingestion.chunker import Chunk
from .base import VectorStore
from .scores import chroma_distance_to_similarity


def _authorized(chunk: Chunk, owner_id: str, tenant_id: str, trusted_only: bool) -> bool:
    if trusted_only and not chunk.trusted:
        return False
    if chunk.owner_id:
        if not owner_id or chunk.owner_id != owner_id:
            return False
    if chunk.tenant_id:
        if not tenant_id or chunk.tenant_id != tenant_id:
            return False
    return True


class ChromaStore(VectorStore):
    def __init__(self, settings):
        import chromadb

        self.settings = settings
        path = str(settings.vector_db_path)
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=settings.vectorstore_collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if not chunks:
            return
        metas = []
        for c in chunks:
            d = c.to_dict()
            d["trusted"] = 1 if c.trusted else 0
            metas.append(d)
        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings.tolist(),
            documents=[c.text for c in chunks],
            metadatas=metas,
        )

    def search(
        self,
        query_embedding: np.ndarray,
        k: int,
        *,
        owner_id: str = "",
        tenant_id: str = "",
        trusted_only: bool = True,
    ) -> list[tuple[Chunk, float]]:
        fetch = max(k * 4, k)
        res = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=fetch,
        )
        results: list[tuple[Chunk, float]] = []
        ids = (res.get("ids") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for i, _cid in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            if "trusted" in meta:
                meta["trusted"] = bool(int(meta["trusted"]))
            dist = dists[i] if i < len(dists) else 1.0
            chunk = Chunk.from_dict(meta)
            if not _authorized(chunk, owner_id, tenant_id, trusted_only):
                continue
            results.append((chunk, chroma_distance_to_similarity(dist)))
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
        data = self.collection.get(include=["metadatas"])
        out = []
        for m in data.get("metadatas") or []:
            if "trusted" in m:
                m["trusted"] = bool(int(m["trusted"]))
            chunk = Chunk.from_dict(m)
            if _authorized(chunk, owner_id, tenant_id, trusted_only):
                out.append(chunk)
        return out

    def count(self) -> int:
        return self.collection.count()

    def delete_source(self, source_id: str) -> None:
        self.collection.delete(where={"source_id": source_id})

    def health_check(self) -> bool:
        try:
            self.collection.count()
            return True
        except Exception:  # noqa: BLE001
            return False