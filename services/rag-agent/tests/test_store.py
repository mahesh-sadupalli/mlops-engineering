from agent.store import find_docs, ingest, search


def test_find_docs_skips_vendored_dirs(docs_dir):
    found = [p.relative_to(docs_dir).as_posix() for p in find_docs(docs_dir)]
    assert found == ["README.md", "k8s/hpa.md"]


def test_ingest_is_idempotent(collection, docs_dir):
    before = collection.count()
    ingest(collection, [docs_dir])
    assert collection.count() == before == 3


def test_search_returns_cited_chunks(collection):
    hits = search(collection, "HPA scales pods CPU", k=1)
    assert hits[0]["source"] == "k8s/hpa.md"
    assert hits[0]["heading"] == "Autoscaling"
    assert 0 < hits[0]["score"] <= 1
