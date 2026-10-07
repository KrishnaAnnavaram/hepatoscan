"""The curated knowledge base: Markdown documents with a front-matter title and source."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

KB_DIR = Path(__file__).with_name("kb")


@dataclass(frozen=True)
class Passage:
    pid: str  # "<document stem>#<paragraph number>"
    title: str
    source: str
    text: str


def parse_document(text: str, stem: str) -> list[Passage]:
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, flags=re.S)
    if not m:
        raise ValueError(f"{stem}: missing front-matter")
    meta = dict(line.split(":", 1) for line in m.group(1).splitlines() if ":" in line)
    meta = {k.strip(): v.strip() for k, v in meta.items()}
    if "title" not in meta or "source" not in meta:
        raise ValueError(f"{stem}: front-matter needs title and source")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", m.group(2)) if p.strip()]
    return [Passage(f"{stem}#{i}", meta["title"], meta["source"], p) for i, p in enumerate(paragraphs, 1)]


def load_kb(directory: Path = KB_DIR) -> list[Passage]:
    passages: list[Passage] = []
    for path in sorted(directory.glob("*.md")):
        passages.extend(parse_document(path.read_text(encoding="utf-8"), path.stem))
    if not passages:
        raise ValueError(f"no knowledge base documents in {directory}")
    return passages
