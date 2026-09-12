"""The leakage guard must reach indexes that already exist on disk.

Round 1 repaired `_golden_set_exclusions()` so it actually excludes the golden
set from the RAG corpus. Round 2 found that the repair was conditional:
`_ensure_seed_data()` returned early whenever the persisted collection was
non-empty, and chroma's data/chroma_db directory survives a `git pull`. So on
every machine that had ever built an index -- including the checkout the project
was developed in -- the pre-fix index was reused verbatim and the eval kept
retrieving the golden set's own reference answers, while the report claimed the
guard was in force.

Measured on a simulated pre-fix index: 121 excluded source ids and 153 golden
reference replies were still retrievable after re-opening with the fixed code.

These tests are slow (they build real embeddings) but they are the only place
that proves the fix applies to an index it did not create.
"""

import json

import chromadb
import pytest

from src.config import EMBEDDING_MODEL_NAME, KAGGLE_PAIRS_PATH
from src.data.heuristic_intent import heuristic_intent
from src.drafting.vector_store import HistoricalVectorStore, _golden_set_exclusions

pytestmark = pytest.mark.skipif(
    not KAGGLE_PAIRS_PATH.exists(),
    reason="needs the real Kaggle corpus; the leakage question is meaningless without it",
)

COLLECTION = "apple_support_resolutions"
N_ROWS = 400


def _build_a_pre_fix_index(path: str) -> int:
    """Writes an index the way the code did BEFORE any leakage guard existed:
    straight into chroma, no exclusions, no build stamp."""
    from src.embeddings import get_encoder

    rows = []
    with open(KAGGLE_PAIRS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= N_ROWS:
                break

    encoder = get_encoder(EMBEDDING_MODEL_NAME)
    vectors = encoder.encode([r["customer_text"] for r in rows], convert_to_numpy=True, normalize_embeddings=True)
    client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})
    collection.upsert(
        ids=[r["tweet_id"] for r in rows],
        documents=[r["customer_text"] for r in rows],
        metadatas=[{"agent_reply": r["agent_reply"], "intent": heuristic_intent(r["customer_text"])} for r in rows],
        embeddings=[v.tolist() for v in vectors],
    )
    return collection.count()


def test_a_pre_fix_index_is_detected_and_rebuilt(tmp_path, caplog):
    path = str(tmp_path / "chroma")
    assert _build_a_pre_fix_index(path) == N_ROWS

    excluded_ids, excluded_texts, excluded_replies = _golden_set_exclusions()
    assert excluded_ids, "the exclusion set is empty -- this test would pass vacuously"

    with caplog.at_level("WARNING"):
        store = HistoricalVectorStore(persist_dir=path)

    indexed = store.collection.get(include=["documents", "metadatas"])
    leaked_ids = set(indexed["ids"]) & excluded_ids
    leaked_texts = [d for d in indexed["documents"] if (d or "").strip() in excluded_texts]
    leaked_replies = [m for m in indexed["metadatas"] if (m.get("agent_reply") or "").strip() in excluded_replies]

    assert leaked_ids == set(), f"{len(leaked_ids)} excluded source ids survived the rebuild"
    assert leaked_texts == [], f"{len(leaked_texts)} golden customer texts survived the rebuild"
    assert leaked_replies == [], f"{len(leaked_replies)} golden reference replies are still retrievable"
    assert "Rebuilding" in caplog.text, "the rebuild must be announced, not silent"


def test_a_current_index_is_not_rebuilt_on_every_start(tmp_path, caplog):
    """The other half: rebuilding costs a full re-embed of the corpus, so a
    correctly-stamped index must be reused. A fingerprint that never matches
    would turn every process start into a multi-minute rebuild."""
    path = str(tmp_path / "chroma")
    first = HistoricalVectorStore(persist_dir=path)
    count = first.collection.count()
    assert count > 0

    with caplog.at_level("WARNING"):
        second = HistoricalVectorStore(persist_dir=path)

    assert second.collection.count() == count
    assert "Rebuilding" not in caplog.text


def test_cosine_distance_survives_the_stamp(tmp_path):
    """`collection.modify()` rejects any metadata containing "hnsw:space", so the
    stamp is written without it and the key disappears from the metadata record.
    The distance function is fixed inside the index at creation and is unaffected
    -- asserted here rather than assumed, because that is exactly the kind of
    claim that rots silently.

    Two orthogonal unit vectors: cosine distance is 1.0, L2 would be ~1.414.
    """
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    collection = client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})
    collection.add(ids=["a"], documents=["x"], embeddings=[[1.0, 0.0, 0.0]])
    collection.modify(metadata={"corpus_fingerprint": "stamp"})

    reopened = chromadb.PersistentClient(path=str(tmp_path / "chroma")).get_or_create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    distance = reopened.query(query_embeddings=[[0.0, 1.0, 0.0]], n_results=1, include=["distances"])["distances"][0][0]
    assert distance == pytest.approx(1.0, abs=1e-4)
