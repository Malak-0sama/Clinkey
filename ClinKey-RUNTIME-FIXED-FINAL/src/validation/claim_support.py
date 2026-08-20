"""Claim → evidence mapping: a citation must support that claim, not just exist."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..retrieval.models import Evidence
from .medical import MedicalSafetyValidator

_SENT = re.compile(r"(?<=[.!?])\s+")
_CITE = re.compile(r"\[(\d{1,2})\]")
# Citation marker(s) sitting at the very front of a split-off segment
# (optionally after a bullet). Because _SENT splits right after the
# terminal punctuation and *before* the citation marker, a marker like
# "[1]" in "...asymptomatic. [1]\n• Screening..." ends up as the leading
# token of the NEXT segment even though it supports the PREVIOUS sentence.
_LEADING_CITE = re.compile(r"^[\s•\-\u2022\*]*((?:\[\d{1,2}\]\s*)+)")
_BULLET_PREFIX = re.compile(r"^[•\-\u2022\*]\s*")


@dataclass
class ClaimMap:
    text: str
    cite_nums: list[int]
    supported: bool
    issues: list[str] = field(default_factory=list)


def extract_claims(answer: str) -> list[ClaimMap]:
    """Split an answer into claims and attach citation markers to the
    sentence they actually support.

    The sentence splitter (``_SENT``) cuts right after terminal punctuation,
    so a trailing citation like "...asymptomatic. [1]" lands at the START of
    the following split segment instead of the END of the one it belongs to.
    We detect such leading citation markers and reattach them to the
    previously built claim (the sentence that actually ends right before
    them), rather than to whatever claim gets built next.
    """
    claims: list[ClaimMap] = []
    for raw in _SENT.split(answer or ""):
        s = raw.strip()
        if not s:
            continue

        lead_nums, remainder = _split_leading_citations(s)
        remainder = _BULLET_PREFIX.sub("", remainder.strip()).strip()
        own_nums = [int(n) for n in _CITE.findall(remainder)]
        text_only = _CITE.sub("", remainder).strip()

        if lead_nums:
            if claims:
                # These citations were split onto the front of this segment
                # but support the claim that precedes it.
                claims[-1].cite_nums.extend(lead_nums)
            elif not text_only:
                # No earlier claim to attach to and nothing else in this
                # segment - there is nothing meaningful left to keep.
                continue
            else:
                # Leading citation on the very first segment of the answer:
                # keep it with the only claim we can build from it.
                own_nums = lead_nums + own_nums

        if not text_only:
            # Segment was citation-marker-only (already merged above) or a
            # bare bullet with no content.
            continue

        claims.append(ClaimMap(text=s, cite_nums=own_nums, supported=True))
    return claims


def _split_leading_citations(s: str) -> tuple[list[int], str]:
    """Return (leading_citation_numbers, remainder_after_them)."""
    m = _LEADING_CITE.match(s)
    if not m:
        return [], s
    nums = [int(n) for n in _CITE.findall(m.group(1))]
    return nums, s[m.end():]


class ClaimCitationValidator:
    def validate(self, answer: str, evidence: list[Evidence]) -> dict:
        claims = extract_claims(answer)
        med = MedicalSafetyValidator()
        bad: list[str] = []
        n = len(evidence)
        for cl in claims:
            if any(i < 1 or i > n for i in cl.cite_nums):
                cl.supported = False
                cl.issues.append("bad_index")
                bad.append(cl.text)
                continue
            if not cl.cite_nums:
                # material medical sentence without a cite is allowed only if
                # the whole corpus supports it (extractive bullets include [n])
                continue
            cited = [evidence[i - 1] for i in cl.cite_nums]
            check = med.validate(re.sub(_CITE, "", cl.text), cited)
            # topic: claim tokens vs cited chunk
            ct = set(re.findall(r"[a-z0-9]{5,}", cl.text.lower()))
            et = set(re.findall(r"[a-z0-9]{5,}", " ".join(e.chunk.text.lower() for e in cited)))
            if ct and len(ct & et) / max(1, len(ct)) < 0.15:
                cl.supported = False
                cl.issues.append("citation_topic_mismatch")
                bad.append(cl.text)
            elif not check.ok and any("negation" in i or "strength" in i or "number" in i for i in check.issues):
                cl.supported = False
                cl.issues.extend(check.issues)
                bad.append(cl.text)
        return {
            "ok": not bad,
            "unsupported": bad[:8],
            "status": "OK" if not bad else "CITATION_FAILURE",
        }