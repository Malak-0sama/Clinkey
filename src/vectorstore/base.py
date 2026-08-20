"""VectorStore interface."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..ingestion.chunker import Chunk


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None: ...

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        k: int,
        *,
        owner_id: str = "",
        tenant_id: str = "",
        trusted_only: bool = True,
    ) -> list[tuple[Chunk, float]]: ...

    @abstractmethod
    def all_chunks(
        self,
        *,
        owner_id: str = "",
        tenant_id: str = "",
        trusted_only: bool = False,
    ) -> list[Chunk]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def delete_source(self, source_id: str) -> None: ...

    @abstractmethod
    def health_check(self) -> bool: ...