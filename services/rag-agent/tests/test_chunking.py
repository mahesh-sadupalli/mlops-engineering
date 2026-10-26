from agent.chunking import chunk_markdown

DOC = """# Title

Intro paragraph.

## Serving

FastAPI loads the model.

```bash
# not a heading
make run
```

## Drift

Mean shift and std ratio.
"""


def test_splits_on_headings():
    chunks = chunk_markdown(DOC, "README.md")
    assert [c.heading for c in chunks] == ["Title", "Serving", "Drift"]
    assert all(c.source == "README.md" for c in chunks)


def test_code_fence_comment_is_not_a_heading():
    chunks = chunk_markdown(DOC, "README.md")
    serving = next(c for c in chunks if c.heading == "Serving")
    assert "# not a heading" in serving.text
    assert "make run" in serving.text


def test_packs_long_sections_under_limit():
    long_doc = "## Big\n\n" + "\n\n".join(["word " * 50] * 10)
    chunks = chunk_markdown(long_doc, "big.md", max_chars=600)
    assert len(chunks) > 1
    assert all(len(c.text) <= 600 for c in chunks)


def test_chunk_ids_are_unique():
    chunks = chunk_markdown(DOC, "README.md")
    assert len({c.id for c in chunks}) == len(chunks)
