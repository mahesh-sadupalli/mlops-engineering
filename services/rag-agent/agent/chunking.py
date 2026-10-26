"""Split markdown into heading-aware chunks for retrieval."""

import hashlib
import re
from dataclasses import dataclass

HEADING = re.compile(r"^(#{1,4})\s+(.*)")


@dataclass
class Chunk:
    text: str
    source: str
    heading: str

    @property
    def id(self) -> str:
        digest = hashlib.sha1(self.text.encode()).hexdigest()[:8]
        return f"{self.source}#{self.heading}#{digest}"


def chunk_markdown(text: str, source: str, max_chars: int = 1200) -> list[Chunk]:
    """Split on headings, then pack paragraphs up to max_chars.

    Each chunk remembers the nearest heading so answers can cite
    a precise section, not just a file. Fenced code blocks are never split.
    """
    sections: list[tuple[str, list[str]]] = [("", [])]
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        match = None if in_fence else HEADING.match(line)
        if match:
            sections.append((match.group(2).strip(), [line]))
        else:
            sections[-1][1].append(line)

    chunks = []
    for heading, lines in sections:
        for block in _pack(_paragraphs(lines), max_chars):
            if block.strip():
                chunks.append(Chunk(text=block.strip(), source=source, heading=heading))
    return chunks


def _paragraphs(lines: list[str]) -> list[str]:
    paras, current, in_fence = [], [], False
    for line in lines:
        if line.strip().startswith("```"):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if current:
                paras.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:
        paras.append("\n".join(current))
    return paras


def _pack(paras: list[str], max_chars: int) -> list[str]:
    blocks, current = [], ""
    for para in paras:
        if current and len(current) + len(para) + 2 > max_chars:
            blocks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        blocks.append(current)
    return blocks
