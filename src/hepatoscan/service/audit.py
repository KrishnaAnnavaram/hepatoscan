"""Append-only audit log. It records what happened, never the image or the client file name."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path


class AuditLog:
    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self.memory: list[dict] = []

    def write(self, event: str, **fields) -> dict:
        record = {"ts": round(time.time(), 3), "event": event, **fields}
        with self._lock:
            self.memory.append(record)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, sort_keys=True) + "\n")
        return record


def content_id(data: bytes) -> str:
    """A short, non-reversible id of the uploaded bytes (first 16 hex digits of SHA-256)."""
    return hashlib.sha256(data).hexdigest()[:16]
