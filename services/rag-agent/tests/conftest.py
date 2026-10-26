import hashlib
import uuid

import chromadb
import pytest
from chromadb import Documents, EmbeddingFunction, Embeddings

from agent.store import ingest


class BagOfWordsEmbedding(EmbeddingFunction):
    """Deterministic, offline embedding so tests never download a model."""

    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        vectors = []
        for text in input:
            vec = [0.0] * 64
            for word in text.lower().split():
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
            vectors.append(vec)
        return vectors

    @staticmethod
    def name() -> str:
        return "bag-of-words-test"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "BagOfWordsEmbedding":
        return BagOfWordsEmbedding()


@pytest.fixture
def docs_dir(tmp_path):
    (tmp_path / "README.md").write_text(
        "# Serving\n\nFastAPI loads the gradient boosting model at startup.\n\n"
        "## Drift\n\nDrift detection compares mean shift against training baselines.\n"
    )
    (tmp_path / "k8s").mkdir()
    (tmp_path / "k8s" / "hpa.md").write_text(
        "# Autoscaling\n\nThe HPA scales pods when CPU passes 70 percent.\n"
    )
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "ignored.md").write_text("# Ignored\n\nvendored file\n")
    return tmp_path


@pytest.fixture
def collection(docs_dir):
    client = chromadb.EphemeralClient()
    col = client.create_collection(
        f"test-{uuid.uuid4().hex[:8]}",
        embedding_function=BagOfWordsEmbedding(),
        metadata={"hnsw:space": "cosine"},
    )
    ingest(col, [docs_dir])
    return col
