"""Deterministic chunker for corpus/policy/*.md.

Splits each doc on its `##` headings, one chunk per section, in document
order - the scheme documented in corpus/README.md. Chunk ids are derived
purely from filename + section order, never from content, so appending a
new trailing `##` section never renumbers an existing chunk, and this
module must never be changed to chunk any other way without also updating
that scheme doc.
"""

from __future__ import annotations

import re
from pathlib import Path

DEFAULT_POLICY_DIR = Path(__file__).resolve().parent.parent / "corpus" / "policy"

_SECTION_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)


def chunk_markdown(text: str, slug: str) -> list[dict]:
    """Split one policy doc's raw markdown into evidence-ready chunks.

    Each chunk's content is its heading (without the `##` marker) followed
    by the section body, so both the retrieval embedding and whatever cites
    the chunk later see the heading's context, not just the body text.
    """
    matches = list(_SECTION_RE.finditer(text))
    chunks = []
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        content = f"{heading}\n\n{body}" if body else heading
        chunks.append(
            {
                "evidence_id": f"doc:{slug}#c{i + 1}",
                "content": content,
            }
        )
    return chunks


def load_policy_chunks(policy_dir: Path | str = DEFAULT_POLICY_DIR) -> list[dict]:
    """Load and chunk every *.md file in policy_dir, sorted by filename."""
    policy_dir = Path(policy_dir)
    chunks = []
    for path in sorted(policy_dir.glob("*.md")):
        slug = path.stem
        text = path.read_text(encoding="utf-8")
        chunks.extend(chunk_markdown(text, slug))
    return chunks
