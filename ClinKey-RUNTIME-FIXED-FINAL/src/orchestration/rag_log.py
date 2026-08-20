"""Privacy-safe structured RAG decision logs. Never logs PHI or secrets."""
from __future__ import annotations

import json
import logging

_log = logging.getLogger("clinkey.rag")


def log_decision(payload: dict) -> None:
    safe = {
        k: payload.get(k)
        for k in (
            "request_id",
            "user_id",
            "retrieval_count",
            "top_score",
            "confidence",
            "threshold",
            "selected_chunk_ids",
            "abstained",
            "generation_called",
            "elapsed_ms",
            "error",
        )
    }
    _log.info("rag_decision %s", json.dumps(safe, default=str))