"""Framework-free request handlers. The FastAPI app and the tests call these functions.

Rules:

* A request needs the bearer token from ``HEPATOSCAN_API_TOKEN``. If the token
  is not set, the service refuses every protected request.
* An upload is processed in memory. Nothing is written to a public folder,
  and nothing of the image stays after the response.
* The audit log gets a content id, the size and the result summary. It never
  gets the client file name or the image.
"""
from __future__ import annotations

import base64
import hmac
from dataclasses import dataclass

from ..assistant.chat import Assistant
from ..overlay import overlay_rgb
from ..segmenter import Segmenter, segment
from .audit import AuditLog, content_id
from .validation import UploadError, validate_upload


class AuthError(PermissionError):
    pass


class NotConfigured(RuntimeError):
    pass


def check_token(expected: str | None, authorization: str | None) -> None:
    if not expected:
        raise NotConfigured("the service has no HEPATOSCAN_API_TOKEN; protected endpoints are off")
    prefix = "Bearer "
    if not authorization or not authorization.startswith(prefix):
        raise AuthError("missing bearer token")
    if not hmac.compare_digest(authorization[len(prefix):].encode(), expected.encode()):
        raise AuthError("wrong token")


@dataclass
class Service:
    segmenter: Segmenter
    assistant: Assistant
    audit: AuditLog
    api_token: str | None
    max_upload_bytes: int

    def segment_upload(self, data: bytes, filename: str | None, authorization: str | None,
                       with_overlay: bool = True) -> dict:
        check_token(self.api_token, authorization)
        cid = content_id(data) if data else "empty"
        try:
            vol = validate_upload(data, filename, self.max_upload_bytes)
        except UploadError as exc:
            self.audit.write("segment_rejected", content_id=cid, bytes=len(data), reason=str(exc))
            raise
        mask, summary = segment(self.segmenter, vol)
        out = {"summary": summary.as_dict()}
        if with_overlay:
            try:
                from ..overlay import to_png

                out["overlay_png_base64"] = base64.b64encode(to_png(overlay_rgb(vol.image, mask))).decode()
            except ImportError:
                out["overlay_png_base64"] = None
        self.audit.write("segment_ok", content_id=cid, bytes=len(data), summary=summary.as_dict())
        return out

    def chat(self, payload: dict, authorization: str | None) -> dict:
        check_token(self.api_token, authorization)
        reply = self.assistant.ask(payload.get("question", ""), session_id=payload.get("session_id"),
                                   summary_text=payload.get("summary_text"))
        self.audit.write("chat", session=reply.session_id[:8], escalated=reply.escalated, refused=reply.refused,
                         citations=len(reply.citations))
        return reply.as_dict()

    def end_chat(self, session_id: str, authorization: str | None) -> dict:
        check_token(self.api_token, authorization)
        return {"deleted": self.assistant.sessions.delete(session_id)}
