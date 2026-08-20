"""Hybrid retrieval: semantic + BM25 fused with Reciprocal Rank Fusion (k=60).

Authorization is applied *before* ranking.
"""
from __future__ import annotations

from ..config.settings import get_settings
from ..vectorstore.base import VectorStore
from .bm25 import BM25
from .errors import UnauthorizedRetrievalError
from .models import Evidence, RetrievalResult, clamp01
from .reranker import Reranker
from .rrf import rrf_normalize, rrf_score
from .entity import entity_score
from .support import filter_supported
from .semantic import SemanticSearch


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder):
        self.settings = get_settings()
        self.store = store
        self.semantic = SemanticSearch(store, embedder)
        self.reranker = Reranker()

    def retrieve(
        self,
        query: str,
        *,
        owner_id: str = "",
        tenant_id: str = "",
        trusted_only: bool = True,
        require_identity: bool | None = None,
        allowed_source_ids: list[str] | None = None,
        support_query: str | None = None,
    ) -> RetrievalResult:
        evaluation_query = support_query or query
        req = (
            bool(self.settings.get("retrieval_require_identity", False))
            if require_identity is None
            else bool(require_identity)
        )
        if req and (not owner_id or not tenant_id):
            raise UnauthorizedRetrievalError(
                "user_id and tenant_id are required for retrieval when "
                "retrieval.require_identity is enabled"
            )

        top_k = int(self.settings.get("retrieval_candidate_k", self.settings.get("retrieval_top_k", 15)))
        max_ctx = int(self.settings.get("retrieval_max_context_chunks", 5))
        rrf_k = int(self.settings.get("retrieval_rrf_k", 60))
        chunks = self.store.all_chunks(
            owner_id=owner_id, tenant_id=tenant_id, trusted_only=trusted_only
        )
        if allowed_source_ids:
            allow = set(allowed_source_ids)
            chunks = [c for c in chunks if c.source_id in allow]
        if not chunks:
            return RetrievalResult(
                evidence=[],
                confidence="INSUFFICIENT_EVIDENCE",
                confidence_score=0.0,
                sufficient=False,
                query=evaluation_query,
            )

        semantic_hits = self.semantic.search(
            query,
            top_k,
            owner_id=owner_id,
            tenant_id=tenant_id,
            trusted_only=trusted_only,
        )
        if allowed_source_ids:
            allow = set(allowed_source_ids)
            semantic_hits = [(c, s) for c, s in semantic_hits if c.source_id in allow]
        bm25 = BM25([c.text for c in chunks])
        bm25_scores = bm25.scores(query)

        sem_rank = {
            chunk.chunk_id: i + 1
            for i, (chunk, _s) in enumerate(
                sorted(semantic_hits, key=lambda x: -x[1])
            )
        }
        bm_order = sorted(range(len(chunks)), key=lambda i: -bm25_scores[i])
        bm_rank = {chunks[i].chunk_id: r + 1 for r, i in enumerate(bm_order)}
        sem_score = {chunk.chunk_id: sim for chunk, sim in semantic_hits}

        combined: dict[str, Evidence] = {}
        for c in chunks:
            ranks = []
            if c.chunk_id in sem_rank:
                ranks.append(sem_rank[c.chunk_id])
            if c.chunk_id in bm_rank:
                ranks.append(bm_rank[c.chunk_id])
            if not ranks:
                continue
            ev = Evidence(chunk=c)
            ev.semantic_score = float(sem_score.get(c.chunk_id, 0.0))
            ev.bm25_score = float(bm25_scores[chunks.index(c)])
            ent = entity_score(query, c)
            ranks = []
            if c.chunk_id in sem_rank:
                ranks.append(sem_rank[c.chunk_id])
            if c.chunk_id in bm_rank:
                ranks.append(bm_rank[c.chunk_id])
            if ent > 0:
                # treat entity hit as a synthetic top-ish rank
                ranks.append(1 if ent >= 1.0 else 3)
            if not ranks:
                continue
            raw = rrf_score(ranks, k=rrf_k)
            ev.rrf_score = raw
            ev.combined_score = rrf_normalize(raw, n_modalities=3 if ent else 2, k=rrf_k)
            combined[c.chunk_id] = ev

        pool = sorted(combined.values(), key=lambda e: -e.combined_score)
        pool = self._dedup(pool)[:top_k]

        if bool(self.settings.get("retrieval_reranker_enabled", True)):
            reranked = self.reranker.rerank(evaluation_query, pool, top_k)
        else:
            reranked = pool
            for i, e in enumerate(reranked):
                e.rank = i + 1
                e.rerank_score = e.combined_score
                e.raw_rerank_score = e.combined_score

        # Always apply the calibrated heuristic floor. CE is not on this path
        # unless backend=cross_encoder AND calibrated (then use CE floor).
        if self.reranker.uses_cross_encoder():
            thr = float(self.settings.get("retrieval_cross_encoder_min_rerank", 0.50))
        else:
            thr = float(self.settings.get("retrieval_heuristic_min_rerank", 0.12))
        min_sup = float(self.settings.get("retrieval_min_support_score", 0.35))
        kept = [e for e in reranked if e.rerank_score >= thr]
        kept = filter_supported(evaluation_query, kept, min_support=min_sup)
        kept = self._dedup(kept)[:max_ctx]

        for i, e in enumerate(kept):
            e.rank = i + 1
            e.relevance_score = e.rerank_score

        return RetrievalResult(
            evidence=kept,
            confidence="PENDING" if kept else "INSUFFICIENT_EVIDENCE",
            confidence_score=max((e.support_score for e in kept), default=0.0),
            query=evaluation_query,
            sufficient=bool(kept),
            generation_allowed=bool(kept),
            status="OK" if kept else "INSUFFICIENT_EVIDENCE",
        )

    def retrieve_many(
        self,
        queries: list[str],
        *,
        support_query: str,
        owner_id: str = "",
        tenant_id: str = "",
        trusted_only: bool = True,
        require_identity: bool | None = None,
        allowed_source_ids: list[str] | None = None,
    ) -> RetrievalResult:
        merged: dict[str, Evidence] = {}
        for query in queries[:3]:
            result = self.retrieve(
                query,
                owner_id=owner_id,
                tenant_id=tenant_id,
                trusted_only=trusted_only,
                require_identity=require_identity,
                allowed_source_ids=allowed_source_ids,
                support_query=support_query,
            )
            for evidence in result.evidence:
                current = merged.get(evidence.chunk.chunk_id)
                if current is None:
                    merged[evidence.chunk.chunk_id] = evidence
                    continue
                current.semantic_score = max(current.semantic_score, evidence.semantic_score)
                current.bm25_score = max(current.bm25_score, evidence.bm25_score)
                current.rrf_score = max(current.rrf_score, evidence.rrf_score)
                current.combined_score = max(current.combined_score, evidence.combined_score)
                current.support_score = max(current.support_score, evidence.support_score)
        if not merged:
            return RetrievalResult(
                evidence=[],
                confidence="INSUFFICIENT_EVIDENCE",
                confidence_score=0.0,
                sufficient=False,
                query=support_query,
                generation_allowed=False,
                status="INSUFFICIENT_EVIDENCE",
            )
        max_ctx = int(self.settings.get("retrieval_max_context_chunks", 5))
        top_k = int(self.settings.get("retrieval_candidate_k", 15))
        reranked = self.reranker.rerank(support_query, list(merged.values()), top_k)
        if self.reranker.uses_cross_encoder():
            threshold = float(self.settings.get("retrieval_cross_encoder_min_rerank", 0.50))
        else:
            threshold = float(self.settings.get("retrieval_heuristic_min_rerank", 0.12))
        min_support = float(self.settings.get("retrieval_min_support_score", 0.35))
        kept = [e for e in reranked if e.rerank_score >= threshold]
        kept = filter_supported(support_query, kept, min_support=min_support)
        kept = self._dedup(kept)[:max_ctx]
        for index, evidence in enumerate(kept):
            evidence.rank = index + 1
            evidence.relevance_score = evidence.rerank_score
        return RetrievalResult(
            evidence=kept,
            confidence="PENDING" if kept else "INSUFFICIENT_EVIDENCE",
            confidence_score=max((e.support_score for e in kept), default=0.0),
            query=support_query,
            sufficient=bool(kept),
            generation_allowed=bool(kept),
            status="OK" if kept else "INSUFFICIENT_EVIDENCE",
        )

    @staticmethod
    def _dedup(pool: list[Evidence]) -> list[Evidence]:
        seen: set[str] = set()
        out: list[Evidence] = []
        for e in pool:
            key = " ".join(e.chunk.text.lower().split())[:160]
            if key in seen:
                continue
            seen.add(key)
            out.append(e)
        return out