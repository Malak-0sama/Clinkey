"""Retrieval data models.

Score semantics (higher is better):

- semantic_score: unit cosine similarity in [0, 1] after store conversion
- bm25_score: raw Okapi BM25 (unbounded)
- rrf_score: raw Reciprocal Rank Fusion
- combined_score: RRF normalized to [0, 1]
- raw_rerank_score: unaltered heuristic [~0,1] or CE logit
- rerank_score: same as raw (kept for callers)
- support_score: heuristic entailment/support in [0, 1]
- relevance_score: post-filter display score (rerank, not overwritten by RRF)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..ingestion.chunker import Chunk


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


@dataclass
class Evidence:
    chunk: Chunk
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    combined_score: float = 0.0
    raw_rerank_score: float = 0.0
    rerank_score: float = 0.0
    support_score: float = 0.0
    relevance_score: float = 0.0
    clinical_relevance_score: float = 0.0
    claim_evaluated: bool = False
    relevant_topic: bool = False
    relevant_intent: bool = False
    relevant_source: bool = True
    directly_supports_claim: bool = False
    required_fact_type: str = ""
    claim_atomic: bool = False
    claim_support_reasons: list[str] = field(default_factory=list)
    rank: int = 0

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "organization": self.chunk.organization,
            "document": self.chunk.document_name,
            "section": self.chunk.section_title,
            "page": self.chunk.page_number,
            "source_url": self.chunk.source_url,
            "source_id": self.chunk.source_id,
            "owner_id": self.chunk.owner_id,
            "trusted": self.chunk.trusted,
            "semantic_score": round(self.semantic_score, 3),
            "bm25_score": round(self.bm25_score, 3),
            "rrf_score": round(self.rrf_score, 4),
            "combined_score": round(self.combined_score, 3),
            "raw_rerank_score": round(self.raw_rerank_score, 4),
            "rerank_score": round(self.rerank_score, 4),
            "support_score": round(self.support_score, 3),
            "relevance_score": round(self.relevance_score, 3),
            "clinical_relevance_score": round(self.clinical_relevance_score, 3),
            "relevant_topic": self.relevant_topic,
            "relevant_intent": self.relevant_intent,
            "relevant_source": self.relevant_source,
            "directly_supports_claim": self.directly_supports_claim,
            "required_fact_type": self.required_fact_type,
            "claim_support_reasons": list(self.claim_support_reasons),
            "rank": self.rank,
            "excerpt": self.chunk.text,
        }


@dataclass
class RetrievalResult:
    evidence: list[Evidence] = field(default_factory=list)
    confidence: str = "LOW"
    confidence_score: float = 0.0
    sufficient: bool = False
    signals: dict = field(default_factory=dict)
    query: str = ""
    generation_allowed: bool = False
    status: str = ""