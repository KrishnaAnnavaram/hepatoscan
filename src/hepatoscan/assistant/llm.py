"""Language-model adapters behind one small interface.

* ``OfflineLLM``: extractive. It quotes the passage sentences that share the
  most words with the question and cites them. No network, no key.
* ``OpenAICompatibleLLM``: any server with an OpenAI-style
  ``/chat/completions`` endpoint (OpenAI, Gemini's OpenAI-compatible endpoint,
  Ollama, vLLM). Standard library HTTP only. The base URL is configuration.
"""
from __future__ import annotations

import json
import re
import urllib.request
from typing import Protocol

from .retriever import tokens


class LLM(Protocol):
    name: str

    def complete(self, messages: list[dict[str, str]]) -> str:  # pragma: no cover - protocol
        ...


class OfflineLLM:
    name = "offline-extractive"

    def complete(self, messages: list[dict[str, str]]) -> str:
        question = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        context = next((m["content"] for m in messages
                        if m["role"] == "system" and m["content"].startswith("PASSAGES\n")), "")
        q = set(tokens(question))
        picks: list[tuple[int, int, str]] = []
        for block in re.findall(r"\[(\d+)\] [^\n]*\n(.*?)(?=\n\[\d+\] |\Z)", context, flags=re.S):
            n, text = int(block[0]), block[1]
            for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
                overlap = len(q & set(tokens(sent)))
                if overlap:
                    picks.append((overlap, n, sent.strip()))
        picks.sort(key=lambda p: (-p[0], p[1]))
        chosen = picks[:3]
        if not chosen:
            return "I do not have information about this in my sources."
        return " ".join(f"{s} [{n}]" for _, n, s in chosen)


class OpenAICompatibleLLM:
    def __init__(self, base_url: str, model: str, api_key: str | None, timeout_s: float = 30.0) -> None:
        if not base_url.startswith(("https://", "http://localhost", "http://127.0.0.1")):
            raise ValueError("the LLM base URL must use https (plain http only for localhost)")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.name = f"openai-compatible:{model}"

    def complete(self, messages: list[dict[str, str]]) -> str:
        body = json.dumps({"model": self.model, "messages": messages, "temperature": 0}).encode()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310 - URL is checked above
            data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"]


class ScriptedLLM:
    """Test double that returns fixed answers in order and records the messages it gets."""

    name = "scripted"

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self.answers.pop(0) if self.answers else ""


def make_llm(provider: str, base_url: str, model: str, api_key: str | None, timeout_s: float) -> LLM:
    if provider == "offline":
        return OfflineLLM()
    if provider == "openai":
        return OpenAICompatibleLLM(base_url, model, api_key, timeout_s)
    raise ValueError(f"unknown LLM provider {provider!r}; use 'offline' or 'openai'")
