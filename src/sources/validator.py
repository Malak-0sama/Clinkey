"""Source validation — the backend is the final authority.

Rejects unknown IDs, disallowed sources, and URLs outside approved domains.
"""
from __future__ import annotations

from dataclasses import dataclass

from .catalog import SourceCatalog
from .models import Source, SourceSelection

# domains the backend trusts for downloads
_ALLOWED_DOMAINS = {
    "who.int",
    "cdc.gov",
    "nice.org.uk",
    "uspreventiveservicestaskforce.org",
    "nih.gov",
    "nlm.nih.gov",           # NCBI / PubMed / PMC literature
    "medlineplus.gov",
    "fda.gov",
    "ema.europa.eu",
    "nhs.uk",
    "cochrane.org",
    "canada.ca",
    "health.gov.au",
    "idsociety.org",         # IDSA practice guidelines
    "heart.org",             # American Heart Association (professional.heart.org)
    "escardio.org",          # European Society of Cardiology
    "diabetesjournals.org",  # ADA Standards of Care (Diabetes Care)
    "gov",
}


@dataclass
class ValidationResult:
    valid: list[Source]          # validated, allowed sources
    rejected: list[dict]         # reasons for rejection


class SourceValidator:
    def __init__(self):
        self.catalog = SourceCatalog()

    def validate(self, selections: list[SourceSelection]) -> ValidationResult:
        valid: list[Source] = []
        rejected: list[dict] = []
        seen: set[str] = set()

        for sel in selections:
            if sel.source_id in seen:
                continue
            seen.add(sel.source_id)

            src = self.catalog.get(sel.source_id)
            if src is None:
                rejected.append({"source_id": sel.source_id, "reason": "unknown source id"})
                continue
            if not src.allowed:
                rejected.append({"source_id": sel.source_id, "reason": "source not allowed"})
                continue
            if not self._domain_allowed(src.official_url):
                rejected.append({"source_id": sel.source_id, "reason": "url domain not allowed"})
                continue
            valid.append(src)

        return ValidationResult(valid=valid, rejected=rejected)

    @staticmethod
    def _domain_allowed(url: str) -> bool:
        from urllib.parse import urlparse

        host = urlparse(url).hostname or ""
        return any(host == d or host.endswith("." + d) for d in _ALLOWED_DOMAINS)