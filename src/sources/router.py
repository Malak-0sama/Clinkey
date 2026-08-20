"""Dynamic Source Router.

Gemini selects from the approved catalog when available; otherwise a
deterministic keyword matcher selects the most relevant allowed sources.
The LLM NEVER downloads files — only the backend does that, after validation.
"""
from __future__ import annotations

import re

from ..llm.client import LLMUnavailable, get_llm
from ..query.analyzer import QueryAnalyzer, QueryAnalysis
from .catalog import SourceCatalog
from .models import SourceSelection


class SourceRouter:
    def __init__(self):
        self.catalog = SourceCatalog()
        self.analyzer = QueryAnalyzer()

    def route(self, english_query: str, max_sources: int = 5) -> list[SourceSelection]:
        # 1) Gemini structured routing
        try:
            llm = get_llm()
            if llm.available:
                sel = self._gemini_route(llm, english_query, max_sources)
                if sel:
                    return sel
        except (LLMUnavailable, Exception):  # noqa: BLE001
            pass

        # 2) deterministic fallback
        return self._keyword_route(english_query, max_sources)

    # ---- deterministic ----
    def _keyword_route(self, english_query: str, max_sources: int) -> list[SourceSelection]:
        analysis: QueryAnalysis = self.analyzer.analyze(english_query)
        ql = english_query.lower()
        organization_patterns = {
            "nice": r"\bNICE\b|\bNational Institute for Health and Care Excellence\b",
            "who": r"\bWHO\b|\bWorld Health Organization\b",
            "cdc": r"\bCDC\b|\bCenters for Disease Control\b",
            "uspstf": r"\bUSPSTF\b",
            "nih": r"\bNIH\b|\bNational Institutes of Health\b",
            "nhs": r"\bNHS\b",
            "fda": r"\bFDA\b|\bFood and Drug Administration\b",
            "ema": r"\bEMA\b|\bEuropean Medicines Agency\b",
            "esc": r"\bESC\b|\bEuropean Society of Cardiology\b",
            "aha": r"\bAHA\b|\bAmerican Heart Association\b",
            "ada": r"\bADA\b|\bAmerican Diabetes Association\b",
            "idsa": r"\bIDSA\b|\bInfectious Diseases Society of America\b",
        }
        requested_organizations = {
            key for key, pattern in organization_patterns.items()
            if re.search(pattern, english_query)
        }
        scored: list[tuple[int, str, str]] = []
        for src in self.catalog.allowed():
            hay = f"{src.topic} {src.title} {src.organization} {src.source_id}".lower()
            if requested_organizations and not any(
                key in src.organization.lower() or key in src.source_id.lower()
                for key in requested_organizations
            ):
                continue
            score = 0
            matched = []
            org_key = re.findall(r"[a-z]+", src.organization.lower())
            org_key = org_key[0] if org_key else ""
            if org_key in organization_patterns and re.search(
                organization_patterns[org_key], english_query
            ):
                score += 10
                matched.append(src.organization)
            for kw in analysis.keywords:
                if kw in hay:
                    score += 2
                    matched.append(kw)
            for topic in analysis.topics:
                if topic in hay:
                    score += 4
                    matched.append(topic)
            if "guideline" in ql and src.resource_type in {"pdf", "page"} and src.topic in analysis.topics:
                score += 2
            if "treatment" in analysis.intents:
                if src.organization.upper() in {"NICE", "AHA", "ESC", "IDSA", "ADA"}:
                    score += 4
                if re.search(r"guideline|guidance|management|treatment", src.title, re.I):
                    score += 3
                if re.search(r"\bstep\s*[123]\b", ql) and src.organization.upper() == "NICE":
                    score += 4
            if "fact sheet" in ql and "fact sheet" in src.title.lower():
                score += 2
            if score > 0:
                scored.append((score, src.source_id, "matched: " + ", ".join(sorted(set(matched)))))
        scored.sort(key=lambda x: -x[0])
        selected = scored[:max_sources]
        if not selected:
            # no topical match -> return all allowed sources as candidates
            for src in self.catalog.allowed():
                selected.append((1, src.source_id, "general fallback"))
            selected = selected[:max_sources]
        return [
            SourceSelection(source_id=sid, relevance="high" if sc >= 5 else "medium", reason=reason)
            for sc, sid, reason in selected
        ]

    # ---- Gemini ----
    @staticmethod
    def _gemini_route(llm, english_query: str, max_sources: int) -> list[SourceSelection] | None:
        catalog = SourceCatalog()
        catalog_json = [s.to_dict() for s in catalog.allowed()]
        system = (
            "You are a medical source router. Given a canonical English medical query and "
            "an approved catalog of official medical guideline sources, select the most "
            "relevant sources (maximum the provided limit). You MUST only choose source_id "
            "values present in the catalog. Respond with JSON only: "
            "{\"selected_sources\": [{\"source_id\": \"...\", \"relevance\": \"high|medium|low\", \"reason\": \"...\"}]}."
        )
        import json

        prompt = (
            f"Query:\n{english_query}\n\n"
            f"Approved catalog (JSON):\n{json.dumps(catalog_json, ensure_ascii=False)}\n\n"
            f"Maximum sources: {max_sources}"
        )
        out = llm.generate_json(prompt, system=system)
        items = out.get("selected_sources", []) if isinstance(out, dict) else out
        if not isinstance(items, list) or not items:
            return None
        selections = []
        for it in items[:max_sources]:
            if isinstance(it, dict) and it.get("source_id"):
                selections.append(
                    SourceSelection(
                        source_id=str(it["source_id"]),
                        relevance=str(it.get("relevance", "medium")),
                        reason=str(it.get("reason", "")),
                    )
                )
        return selections or None