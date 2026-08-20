"""History repository — per-user, session-isolated conversation persistence.

Each authenticated user's chat history is stored in its own file
(`data/history/sessions_<user_id>.json`) so one user's data is never
exposed to another. Existing pre-auth sessions are left in place.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..config.settings import get_settings
from .models import Message, Session

_SAFE_ID = re.compile(r"[^a-zA-Z0-9_-]")


class HistoryRepository:
    def __init__(self, user_id: str = "anonymous"):
        self.settings = get_settings()
        self.user_id = _SAFE_ID.sub("_", user_id) or "anonymous"
        self.dir: Path = self.settings.history_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = self.dir / f"sessions_{self.user_id}.json"

    # ---- load / save ----
    def _load(self) -> dict[str, Session]:
        if not self.file.exists():
            return {}
        try:
            data = json.loads(self.file.read_text(encoding="utf-8"))
            return {sid: Session.from_dict(d) for sid, d in data.items()}
        except (json.JSONDecodeError, KeyError):
            return {}

    def _save(self, sessions: dict[str, Session]) -> None:
        payload = {sid: s.to_dict() for sid, s in sessions.items()}
        self.file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- API ----
    def create_session(self, title: str = "New chat") -> Session:
        sessions = self._load()
        session = Session(title=title, user_id=self.user_id)
        sessions[session.session_id] = session
        self._save(sessions)
        return session

    def get_session(self, session_id: str) -> Session | None:
        return self._load().get(session_id)

    def list_sessions(self) -> list[Session]:
        sessions = self._load()
        return sorted(sessions.values(), key=lambda s: s.created_at, reverse=True)

    def add_message(self, session_id: str, message: Message) -> None:
        sessions = self._load()
        session = sessions.get(session_id)
        if session is None:
            session = Session(session_id=session_id, user_id=self.user_id)
            sessions[session_id] = session
        session.messages.append(message)
        # auto-title from first user message
        if session.title in ("", "New chat") and message.role == "user" and message.original_text:
            session.title = message.original_text[:40] + ("…" if len(message.original_text) > 40 else "")
        self._save(sessions)

    def delete_session(self, session_id: str) -> None:
        sessions = self._load()
        sessions.pop(session_id, None)
        self._save(sessions)

    def rename_session(self, session_id: str, new_title: str) -> None:
        sessions = self._load()
        session = sessions.get(session_id)
        if session is None:
            return
        title = (new_title or "").strip()
        if title:
            session.title = title[:80]
            self._save(sessions)

    def search(self, query: str) -> list[Session]:
        ql = query.lower()
        return [
            s for s in self.list_sessions()
            if ql in s.title.lower()
            or any(ql in m.original_text.lower() or ql in m.localized_answer.lower() for m in s.messages)
        ]