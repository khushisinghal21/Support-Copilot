"""ChromaDB Vector Store for indexing historical @AppleSupport resolution pairs."""

import json
import logging
from typing import List, Dict, Optional
from src.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME, KAGGLE_PAIRS_PATH, GOLDEN_SET_PATH
from src.drafting.historical_data import HISTORICAL_APPLE_RESOLUTIONS
from src.data.heuristic_intent import heuristic_intent

logger = logging.getLogger(__name__)


def _golden_set_source_ids() -> set:
    """Returns the set of original kaggle source_tweet_ids used in the golden
    eval set, so the RAG corpus can exclude them and avoid retrieval leakage
    (the eval set would otherwise be able to retrieve, near-verbatim, the
    exact real reply it's being scored against)."""
    ids = set()
    if not GOLDEN_SET_PATH.exists():
        return ids
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            src_id = d.get("source_tweet_id")
            if src_id:
                ids.add(str(src_id))
    return ids


def load_real_corpus(max_records: int = 800) -> List[Dict[str, str]]:
    """Loads the real Kaggle-extracted @AppleSupport pairs for RAG indexing,
    excluding anything used in the golden evaluation set (leakage guard) and
    re-tagging intent with the independent heuristic rather than trusting the
    tag assigned at ingestion time by the same classifier under evaluation."""
    if not KAGGLE_PAIRS_PATH.exists():
        return []
    excluded = _golden_set_source_ids()
    records: List[Dict[str, str]] = []
    with open(KAGGLE_PAIRS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if str(d.get("tweet_id")) in excluded:
                continue
            records.append({
                "tweet_id": d["tweet_id"],
                "customer_text": d["customer_text"],
                "agent_reply": d["agent_reply"],
                "intent": heuristic_intent(d["customer_text"]),
            })
            if len(records) >= max_records:
                break
    return records


class HistoricalVectorStore:
    """Manages the embedded ChromaDB vector collection for customer support RAG."""

    def __init__(self, persist_dir: Optional[str] = None, collection_name: str = "apple_support_resolutions"):
        # chromadb is imported here, not at module level. A module-level
        # import would make every "import src.server" (which imports this
        # module transitively) pay the full cost of importing
        # torch/chromadb before the ASGI server can even bind its port -- on
        # a low-CPU deploy host that import cost alone was enough to blow
        # past the platform's port-scan timeout, independent of how lazily
        # the objects below are actually constructed.
        #
        # get_encoder() (src/embeddings.py) caches the SentenceTransformer
        # by model name, so this shares the same in-memory model instance
        # with SemanticCentroidClassifier instead of loading a second,
        # redundant copy -- that duplication was a contributor to an
        # out-of-memory crash on Render's 512MB free tier.
        import chromadb
        from src.embeddings import get_encoder

        self.persist_dir = str(persist_dir or CHROMA_PERSIST_DIR)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.encoder = get_encoder(EMBEDDING_MODEL_NAME)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self._ensure_seed_data()

    def _ensure_seed_data(self):
        """Populates the collection if empty. Prefers the real Kaggle-extracted
        corpus (excluding golden-set leakage) over the ~40 hand-written seed
        pairs, which are indexed only as a fallback/supplement when the real
        corpus file isn't present."""
        if self.collection.count() != 0:
            return
        real_records = load_real_corpus()
        if real_records:
            logger.info(f"Initializing vector store with {len(real_records)} real "
                        f"@AppleSupport pairs (golden-set examples excluded)...")
            self.index_records(real_records)
        else:
            logger.warning("No real Kaggle pairs found at KAGGLE_PAIRS_PATH -- "
                            "falling back to the small hand-written seed corpus. "
                            "Run `python -m src.data.ingest_kaggle extract` first "
                            "for real grounding.")
            self.index_records(HISTORICAL_APPLE_RESOLUTIONS)

    def index_records(self, records: List[Dict[str, str]]):
        """Indexes a list of resolution records into ChromaDB."""
        ids = []
        documents = []
        metadatas = []
        embeddings = []

        texts_to_embed = [r["customer_text"] for r in records]
        encoded_vecs = self.encoder.encode(texts_to_embed, convert_to_numpy=True, normalize_embeddings=True)

        for record, vec in zip(records, encoded_vecs):
            ids.append(record["tweet_id"])
            documents.append(record["customer_text"])
            metadatas.append({
                "agent_reply": record["agent_reply"],
                "intent": record.get("intent", "OS_SOFTWARE_TROUBLESHOOTING"),
            })
            embeddings.append(vec.tolist())

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def query(self, query_text: str, intent: Optional[str] = None, top_k: int = 3) -> Dict:
        """Queries the collection for semantically similar historical customer tweets."""
        query_vec = self.encoder.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()

        where_filter = {"intent": intent} if intent and intent != "OUT_OF_SCOPE_AMBIGUOUS" else None

        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        return results
