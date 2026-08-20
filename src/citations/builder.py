"""Citation builder — traceable evidence identifiers.

Citations are evidence identifiers, NOT prose: they are built BEFORE
localization and must not be translated.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..retrieval.models import Evidence


@dataclass
class Citation:
    number: int
    organization: str
    document: str
    section: str
    page: int
    chunk_id: str
    source_url: str

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "organization": self.organization,
            "document": self.document,
            "section": self.section,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "source_url": self.source_url,
        }

    @property
    def label(self) -> str:
        return f"{self.organization} — {self.document}"


class CitationBuilder:
    def build(self, cited_evidence: list[Evidence]) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[str] = set()
        for e in cited_evidence:
            if e.chunk.chunk_id in seen:
                continue
            seen.add(e.chunk.chunk_id)
            citations.append(
                Citation(
                    number=len(citations) + 1,
                    organization=e.chunk.organization,
                    document=e.chunk.document_name,
                    section=e.chunk.section_title,
                    page=e.chunk.page_number,
                    chunk_id=e.chunk.chunk_id,
                    source_url=e.chunk.source_url,
                )
            )
        return citations