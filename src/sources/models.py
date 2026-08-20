"""Source data models."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Source:
    source_id: str
    organization: str
    title: str
    topic: str
    official_url: str
    pdf_url: str | None
    resource_type: str
    publication_date: str
    version: str
    allowed: bool

    @classmethod
    def from_dict(cls, d: dict) -> "Source":
        return cls(
            source_id=d["source_id"],
            organization=d["organization"],
            title=d["title"],
            topic=d.get("topic", ""),
            official_url=d["official_url"],
            pdf_url=d.get("pdf_url"),
            resource_type=d.get("resource_type", "page"),
            publication_date=d.get("publication_date", ""),
            version=d.get("version", ""),
            allowed=bool(d.get("allowed", True)),
        )

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "organization": self.organization,
            "title": self.title,
            "topic": self.topic,
            "official_url": self.official_url,
            "pdf_url": self.pdf_url,
            "resource_type": self.resource_type,
            "publication_date": self.publication_date,
            "version": self.version,
            "allowed": self.allowed,
        }


@dataclass
class SourceSelection:
    source_id: str
    relevance: str
    reason: str

    def to_dict(self) -> dict:
        return {"source_id": self.source_id, "relevance": self.relevance, "reason": self.reason}