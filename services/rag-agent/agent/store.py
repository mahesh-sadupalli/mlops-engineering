"""Chroma-backed vector store: ingest markdown docs, retrieve relevant chunks."""

import argparse
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection

from .chunking import chunk_markdown
from .config import settings

SKIP_DIRS = {"node_modules", "chroma"}


def get_collection() -> Collection:
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return client.get_or_create_collection(
        settings.collection, metadata={"hnsw:space": "cosine"}
    )


def _skipped(part: str) -> bool:
    return part in SKIP_DIRS or part.startswith((".", "_"))


def find_docs(root: Path) -> list[Path]:
    """All markdown under root, ignoring hidden, vendored and build dirs."""
    return sorted(
        p
        for p in root.rglob("*.md")
        if not any(_skipped(part) for part in p.relative_to(root).parts[:-1])
    )


def ingest(collection: Collection, roots: list[Path]) -> int:
    """Chunk every markdown file under roots and upsert into the collection."""
    chunks = []
    for root in roots:
        for path in find_docs(root):
            source = str(path.relative_to(root))
            chunks += chunk_markdown(path.read_text(), source, settings.chunk_size)

    if chunks:
        collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "heading": c.heading} for c in chunks],
        )
    return len(chunks)


def search(collection: Collection, query: str, k: int = settings.top_k) -> list[dict]:
    res = collection.query(query_texts=[query], n_results=k)
    return [
        {
            "text": doc,
            "source": meta["source"],
            "heading": meta["heading"],
            "score": round(1 - dist, 3),
        }
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest markdown docs into Chroma")
    parser.add_argument("roots", nargs="*", type=Path, default=settings.docs_dirs)
    args = parser.parse_args()
    n = ingest(get_collection(), [r.resolve() for r in args.roots])
    print(f"Ingested {n} chunks into '{settings.collection}'")
