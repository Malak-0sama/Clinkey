"""Section-aware chunking.

Produces chunks of ~chunk_size_tokens that respect section/paragraph
boundaries and carry full provenance metadata (page, section, source...).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from ..config.settings import get_settings
from ..sources.models import Source


@dataclass
class Chunk:
    chunk_id: str
    source_id: str
    document_name: str
    organization: str
    page_number: int
    section_title: str
    source_url: str
    publication_date: str
    version: str
    text: str
    owner_id: str = ""
    tenant_id: str = ""
    trusted: bool = True

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "document_name": self.document_name,
            "organization": self.organization,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "source_url": self.source_url,
            "publication_date": self.publication_date,
            "version": self.version,
            "text": self.text,
            "owner_id": self.owner_id,
            "tenant_id": self.tenant_id,
            "trusted": self.trusted,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(
            chunk_id=d["chunk_id"],
            source_id=d["source_id"],
            document_name=d["document_name"],
            organization=d["organization"],
            page_number=int(d.get("page_number", 1)),
            section_title=d.get("section_title", ""),
            source_url=d.get("source_url", ""),
            publication_date=d.get("publication_date", ""),
            version=d.get("version", ""),
            text=d["text"],
            owner_id=d.get("owner_id", ""),
            tenant_id=d.get("tenant_id", ""),
            trusted=bool(d.get("trusted", True)),
        )

    @property
    def cite_label(self) -> str:
        return f"{self.organization} — {self.document_name}"


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").upper()
    return s[:20] or "SEC"


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.33))


class SectionAwareChunker:
    def __init__(self):
        self.settings = get_settings()
        self.target = int(self.settings.ingestion_chunk_size_tokens)
        self.overlap = int(self.settings.ingestion_chunk_overlap_tokens)
        self.min_chunk = int(self.settings.ingestion_min_chunk_tokens)

    def chunk(self, source: Source, text: str, page_offset: int = 0) -> list[Chunk]:
        text = text.strip()
        if not text:
            return []

        sections = self._split_sections(text)
        chunks: list[Chunk] = []
        for section_title, body in sections:
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
            buffer: list[str] = []
            buf_tokens = 0
            for para in paragraphs:
                pt = _estimate_tokens(para)
                if buffer and buf_tokens + pt > self.target:
                    chunks.append(self._make_chunk(source, section_title, buffer, page_offset))
                    # sliding-window overlap (token budget) so procedural steps are not cut
                    tail_parts: list[str] = []
                    tail_tok = 0
                    if self.overlap > 0:
                        for para in reversed(buffer):
                            tail_parts.insert(0, para)
                            tail_tok += _estimate_tokens(para)
                            if tail_tok >= self.overlap:
                                break
                    buffer = tail_parts
                    buf_tokens = tail_tok
                buffer.append(para)
                buf_tokens += pt
            if buffer:
                chunks.append(self._make_chunk(source, section_title, buffer, page_offset))

        # filter tiny chunks
        return [c for c in chunks if _estimate_tokens(c.text) >= self.min_chunk]

    # ---- internals ----
    def _split_sections(self, text: str) -> list[tuple[str, str]]:
        lines = text.splitlines()
        sections: list[tuple[str, str]] = []
        current_title = ""
        current_body: list[str] = []
        for line in lines:
            s = line.strip()
            if self._looks_like_heading(s):
                if current_body or current_title:
                    sections.append((current_title, "\n".join(current_body)))
                current_title = s
                current_body = []
            else:
                current_body.append(line)
        if current_body or current_title:
            sections.append((current_title, "\n".join(current_body)))
        return sections

    @staticmethod
    def _looks_like_heading(line: str) -> bool:
        if not line or len(line) > 80:
            return False
        if re.search(r"[.!?,;:]$", line):
            return False
        if line.isupper() and len(line.split()) <= 8:
            return True
        if re.match(r"^(Chapter|Section|\d+\.\d*)\s", line, re.IGNORECASE):
            return True
        words = line.split()
        return len(words) <= 8 and all(
            w[0].isupper() or w.isdigit() or w.lower() in {"of", "and", "the", "in", "for", "to"} for w in words
        )

    def _make_chunk(self, source: Source, section_title: str, parts: list[str], page_offset: int) -> Chunk:
        body = " ".join(parts).strip()
        header = f"# {source.organization} — {source.title}"
        if section_title:
            header = f"{header} -> ## {section_title}"
        text = f"{header}\n\n{body}"
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8].upper()
        chunk_id = f"{_slug(source.source_id)}_{_slug(section_title)}_{digest}"
        return Chunk(
            chunk_id=chunk_id,
            source_id=source.source_id,
            document_name=source.title,
            organization=source.organization,
            page_number=page_offset + 1,
            section_title=section_title,
            source_url=source.official_url,
            publication_date=source.publication_date,
            version=source.version,
            text=text,
        )