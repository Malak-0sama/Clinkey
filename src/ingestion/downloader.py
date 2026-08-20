"""Source acquisition — downloads ONLY validated sources."""
from __future__ import annotations

import html
import re

import requests

from ..sources.models import Source

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/pdf,application/xhtml+xml,*/*",
}

_TIMEOUT = 25


class DownloadError(Exception):
    pass


class Downloader:
    def __init__(self):
        self.pdf_loader = None  # lazy import to avoid hard dependency at import time

    def fetch(self, source: Source) -> str:
        """Return extracted TEXT for a validated source, or raise DownloadError."""
        url = source.pdf_url or source.official_url
        if not url:
            raise DownloadError("source has no URL")

        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT, allow_redirects=True)
        resp.raise_for_status()
        content = resp.content
        ctype = (resp.headers.get("Content-Type") or "").lower()

        if source.resource_type == "pdf" or "pdf" in ctype or content[:4] == b"%PDF":
            if self.pdf_loader is None:
                from .pdf_loader import PDFLoader

                self.pdf_loader = PDFLoader()
            text = self.pdf_loader.extract_text(content)
            if not text.strip():
                raise DownloadError("PDF extraction produced no text")
            return text

        # HTML page
        return self._html_to_text(content)

    @staticmethod
    def _html_to_text(content: bytes) -> str:
        raw = content.decode("utf-8", errors="ignore")
        raw = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
        raw = re.sub(r"<(nav|header|footer|aside)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
        # prefer main/article when present
        mains = re.findall(r"<(?:main|article)[^>]*>(.*?)</(?:main|article)>", raw, flags=re.S | re.I)
        if mains and sum(len(m) for m in mains) > 200:
            raw = "\n".join(mains)
        raw = re.sub(r"<[^>]+>", "\n", raw)
        raw = html.unescape(raw)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{2,}", "\n\n", raw)
        return raw.strip()