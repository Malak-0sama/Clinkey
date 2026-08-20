"""PDF loading with graceful fallbacks."""
from __future__ import annotations


class PDFLoader:
    def extract_text(self, pdf_bytes: bytes) -> str:
        if not pdf_bytes:
            return ""
        # primary: pypdf
        try:
            import io

            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(pdf_bytes))
            pages = []
            for page in reader.pages:
                try:
                    pages.append(page.extract_text() or "")
                except Exception:  # noqa: BLE001
                    pages.append("")
            text = "\n\n".join(pages)
            if text.strip():
                return text
        except Exception:  # noqa: BLE001
            pass

        # fallback: raw decode (very lossy, last resort)
        try:
            return pdf_bytes.decode("utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            return ""