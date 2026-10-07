"""The patient assistant: retrieval, safety rules, citations and a server-side session history."""
from __future__ import annotations

import re
import secrets
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field

from .kb import Passage, load_kb
from .llm import LLM, OfflineLLM
from .retriever import BM25, Hit
from .safety import ESCALATION, NON_DIAGNOSTIC, has_diagnostic_claim, triage, with_disclaimer

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a liver CT research tool. Rules: "
    "1) Answer only from the numbered PASSAGES and cite them as [n]. "
    "2) If the passages do not answer the question, say that you do not know. "
    "3) Never state or guess a diagnosis, a prognosis, a lesion type or a medicine dose. "
    "4) A scan summary is an automatic measurement, not a diagnosis. "
    "5) Tell the user to discuss results with their care team. Use short, plain sentences."
)
NO_INFO = "I do not have information about this in my sources. Please ask your care team."
MAX_QUESTION_CHARS = 2000


@dataclass
class Session:
    sid: str
    turns: deque = field(default_factory=deque)  # (role, content) pairs
    last_used: float = field(default_factory=time.monotonic)


class SessionStore:
    """In-memory sessions with a turn cap and a time-to-live. Nothing is written to disk or to the browser."""

    def __init__(self, max_exchanges: int = 6, ttl_s: float = 1800.0, max_sessions: int = 1000) -> None:
        self.max_exchanges = max_exchanges
        self.ttl_s = ttl_s
        self.max_sessions = max_sessions
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def _expire(self, now: float) -> None:
        for sid in [s for s, v in self._sessions.items() if now - v.last_used > self.ttl_s]:
            del self._sessions[sid]

    def get(self, sid: str | None) -> Session:
        now = time.monotonic()
        with self._lock:
            self._expire(now)
            if sid and sid in self._sessions:
                sess = self._sessions[sid]
            else:
                if len(self._sessions) >= self.max_sessions:
                    oldest = min(self._sessions.values(), key=lambda s: s.last_used)
                    del self._sessions[oldest.sid]
                sess = Session(secrets.token_urlsafe(16))
                self._sessions[sess.sid] = sess
            sess.last_used = now
            return sess

    def add_exchange(self, sess: Session, question: str, answer: str) -> None:
        with self._lock:
            sess.turns.append(("user", question))
            sess.turns.append(("assistant", answer))
            while len(sess.turns) > 2 * self.max_exchanges:
                sess.turns.popleft()

    def delete(self, sid: str) -> bool:
        with self._lock:
            return self._sessions.pop(sid, None) is not None

    def __len__(self) -> int:
        return len(self._sessions)


@dataclass
class Reply:
    session_id: str
    answer: str
    citations: list[dict]
    escalated: bool = False
    refused: bool = False
    grounded: bool = True
    model: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def valid_markers(answer: str, n_passages: int) -> tuple[str, list[int]]:
    """Remove markers that point to no passage. Return the cleaned answer and the cited numbers."""
    used: list[int] = []

    def repl(m: re.Match) -> str:
        n = int(m.group(1))
        if 1 <= n <= n_passages:
            if n not in used:
                used.append(n)
            return m.group(0)
        return ""

    cleaned = re.sub(r"\[(\d+)\]", repl, answer)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip(), sorted(used)


class Assistant:
    def __init__(self, llm: LLM | None = None, passages: list[Passage] | None = None, max_exchanges: int = 6,
                 ttl_s: float = 1800.0, k: int = 3) -> None:
        self.llm = llm or OfflineLLM()
        self.fallback = OfflineLLM()
        self.retriever = BM25(passages or load_kb())
        self.sessions = SessionStore(max_exchanges, ttl_s)
        self.k = k

    def _messages(self, sess: Session, question: str, hits: list[Hit], summary_text: str | None) -> list[dict[str, str]]:
        block = "\n".join(f"[{i}] {h.passage.title} ({h.passage.source})\n{h.passage.text}" for i, h in enumerate(hits, 1))
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "system", "content": "PASSAGES\n" + block}]
        if summary_text:
            msgs.append({"role": "system", "content": "SCAN SUMMARY (automatic measurement, not a diagnosis): "
                         + summary_text[:500]})
        msgs.extend({"role": role, "content": content} for role, content in sess.turns)
        msgs.append({"role": "user", "content": question})
        return msgs

    def _generate(self, msgs: list[dict[str, str]], n: int) -> tuple[str, list[int], str]:
        for llm in (self.llm, self.fallback):
            try:
                raw = llm.complete(msgs)
            except Exception:  # a provider error must not break the session
                continue
            text, cited = valid_markers(raw, n)
            if cited and not has_diagnostic_claim(text):
                return text, cited, llm.name
        return NO_INFO, [], self.fallback.name

    def ask(self, question: str, session_id: str | None = None, summary_text: str | None = None) -> Reply:
        question = (question or "").strip()
        if not question:
            raise ValueError("empty question")
        if len(question) > MAX_QUESTION_CHARS:
            raise ValueError(f"question is longer than {MAX_QUESTION_CHARS} characters")
        sess = self.sessions.get(session_id)
        tri = triage(question)
        if tri.red_flag:
            answer = with_disclaimer(ESCALATION)
            self.sessions.add_exchange(sess, question, answer)
            return Reply(sess.sid, answer, [], escalated=True, model="rule")
        hits = self.retriever.search(question, k=self.k)
        if not hits:
            answer = with_disclaimer(NON_DIAGNOSTIC + " " + NO_INFO if tri.diagnosis_request else NO_INFO)
            self.sessions.add_exchange(sess, question, answer)
            return Reply(sess.sid, answer, [], refused=tri.diagnosis_request, grounded=False, model="rule")
        text, cited, model = self._generate(self._messages(sess, question, hits, summary_text), len(hits))
        if tri.diagnosis_request:
            text = NON_DIAGNOSTIC + "\n\n" + text
        citations = [{"n": n, "id": hits[n - 1].passage.pid, "title": hits[n - 1].passage.title,
                      "source": hits[n - 1].passage.source} for n in cited]
        if citations:
            text += "\n\nSources:\n" + "\n".join(f"[{c['n']}] {c['title']} ({c['source']})" for c in citations)
        answer = with_disclaimer(text)
        self.sessions.add_exchange(sess, question, answer)
        return Reply(sess.sid, answer, citations, refused=tri.diagnosis_request, grounded=bool(citations), model=model)
