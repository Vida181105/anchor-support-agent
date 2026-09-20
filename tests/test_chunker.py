import glob
import re

from src.chunker import chunk_markdown, load_policy_chunks
from src.evidence import make_evidence

SAMPLE_DOC = """# Sample Policy

## First Section

Body text for the first section.

## Second Section

Body text for the second section,
spanning two lines.

## Third Section

Body text for the third section.
"""


def test_chunk_count_matches_heading_count():
    chunks = chunk_markdown(SAMPLE_DOC, "sample")
    assert len(chunks) == 3


def test_chunk_ids_are_sequential_from_one():
    chunks = chunk_markdown(SAMPLE_DOC, "sample")
    assert [c["evidence_id"] for c in chunks] == [
        "doc:sample#c1",
        "doc:sample#c2",
        "doc:sample#c3",
    ]


def test_chunk_content_includes_heading_and_body():
    chunks = chunk_markdown(SAMPLE_DOC, "sample")
    assert chunks[0]["content"].startswith("First Section")
    assert "Body text for the first section." in chunks[0]["content"]
    assert "##" not in chunks[0]["content"]  # heading marker stripped


def test_appending_a_trailing_section_does_not_renumber_existing_chunks():
    original = chunk_markdown(SAMPLE_DOC, "sample")
    extended_doc = SAMPLE_DOC + "\n## Fourth Section\n\nA new trailing section.\n"
    extended = chunk_markdown(extended_doc, "sample")

    assert len(extended) == 4
    for i in range(3):
        assert original[i] == extended[i]
    assert extended[3]["evidence_id"] == "doc:sample#c4"


def test_chunking_is_deterministic():
    a = chunk_markdown(SAMPLE_DOC, "sample")
    b = chunk_markdown(SAMPLE_DOC, "sample")
    assert a == b


def test_real_policy_corpus_chunk_count_matches_heading_count():
    chunks = load_policy_chunks()
    heading_count = 0
    for path in glob.glob("corpus/policy/*.md"):
        text = open(path).read()
        heading_count += len(re.findall(r"^##\s+", text, re.MULTILINE))
    assert len(chunks) == heading_count
    assert len(chunks) > 0


def test_every_real_chunk_id_is_valid_evidence_id():
    for chunk in load_policy_chunks():
        # make_evidence validates the id format and raises if malformed
        make_evidence(chunk["evidence_id"], chunk["content"], "policy")


def test_real_chunk_ids_are_unique():
    ids = [c["evidence_id"] for c in load_policy_chunks()]
    assert len(ids) == len(set(ids))
