"""ChromaDB Vector Store for indexing historical @AppleSupport resolution pairs."""

import json
import logging

from src.config import (
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL_NAME,
    GOLDEN_SET_PATH,
    KAGGLE_PAIRS_PATH,
    RAG_CORPUS_MAX_RECORDS,
)
from src.data.heuristic_intent import heuristic_intent
from src.drafting.historical_data import HISTORICAL_APPLE_RESOLUTIONS

logger = logging.getLogger(__name__)


def _id_variants(raw: str) -> set[str]:
    """Every corpus id a golden `source_tweet_id` could be referring to.

    WHY THIS IS NOT JUST {raw}
    ---------------------------
    `scripts/finalize_golden_set.py` built each golden row's `tweet_id` as
    f"kaggle_{item['source_tweet_id']}" where `source_tweet_id` ALREADY carried
    that prefix, then copied the result back into `source_tweet_id` (lines 121,
    163, 195). So golden rows carry doubled ids like

        kaggle_kaggle_187962_187961

    while `data/apple_support_kaggle_pairs.jsonl` contains

        kaggle_187962_187961

    Those strings can never be equal, so the leakage guard below excluded
    **nothing** -- while README.md, docs/REPORT.md and CLAUDE.md all claimed it
    worked, and a test asserted it by checking only that both sets were
    non-empty. Collapsing the repeated prefix is what makes the comparison
    actually compare.
    """
    variants = {raw}
    collapsed = raw
    while collapsed.startswith("kaggle_kaggle_"):
        collapsed = collapsed[len("kaggle_") :]
        variants.add(collapsed)
    return variants


def _golden_set_exclusions() -> tuple[set[str], set[str], set[str]]:
    """Returns (source ids, customer texts, reference replies) used by the golden
    eval set, so the RAG corpus can exclude them and avoid retrieval leakage --
    otherwise an eval row can retrieve, verbatim, the exact real reply it is
    being scored against.

    Three keys, and the first one is the one that failed. Matching on id alone is
    what broke here: a formatting change upstream silently turned the guard into
    a no-op and nothing noticed for the life of the project. The other two cannot
    drift that way:

      * exact `customer_text` -- if the same tweet text is in both files, it is
        the same tweet, whatever either file calls it.
      * exact `agent_reply` -- @AppleSupport reuses canned replies across
        different conversations, so a golden row's reference answer can sit in
        the corpus attached to a *different* customer tweet. Retrieving it still
        lets a draft be byte-identical to the reference it is scored against,
        which inflates ROUGE-L without the system having produced anything.
        Measured: excluding by id and customer text alone still left 39 golden
        reference replies reachable in the corpus.

    This deliberately over-excludes: a reply string shared by several
    conversations is removed from the corpus entirely, which costs the retriever
    some genuinely useful (if generic) examples. That is the right direction for
    an evaluation corpus -- a slightly smaller corpus is a cost, a corpus that
    contains the answer key is a broken measurement. The count removed is logged
    and reported.
    """
    ids: set[str] = set()
    texts: set[str] = set()
    replies: set[str] = set()
    if not GOLDEN_SET_PATH.exists():
        return ids, texts, replies
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
                ids |= _id_variants(str(src_id))
            text = (d.get("text") or "").strip()
            if text:
                texts.add(text)
            reply = (d.get("reference_resolution") or "").strip()
            if reply:
                replies.add(reply)
    return ids, texts, replies


def _golden_set_source_ids() -> set:
    """Backwards-compatible accessor for the id part of the exclusion set."""
    ids, _, _ = _golden_set_exclusions()
    return ids


def load_real_corpus(max_records: int = RAG_CORPUS_MAX_RECORDS) -> list[dict[str, str]]:
    """Loads the real Kaggle-extracted @AppleSupport pairs for RAG indexing,
    excluding anything used in the golden evaluation set (leakage guard) and
    re-tagging intent with the independent heuristic rather than trusting the
    tag assigned at ingestion time by the same classifier under evaluation."""
    if not KAGGLE_PAIRS_PATH.exists():
        return []
    excluded_ids, excluded_texts, excluded_replies = _golden_set_exclusions()
    skipped = 0
    records: list[dict[str, str]] = []
    with open(KAGGLE_PAIRS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if (
                str(d.get("tweet_id")) in excluded_ids
                or (d.get("customer_text") or "").strip() in excluded_texts
                or (d.get("agent_reply") or "").strip() in excluded_replies
            ):
                skipped += 1
                continue
            records.append(
                {
                    "tweet_id": d["tweet_id"],
                    "customer_text": d["customer_text"],
                    "agent_reply": d["agent_reply"],
                    "intent": heuristic_intent(d["customer_text"]),
                }
            )
            if len(records) >= max_records:
                break
    # Logged, not silent. A guard that excludes zero rows looks exactly like a
    # guard that is working, which is how the id-format bug survived this long.
    if skipped == 0:
        logger.warning(
            "Leakage guard excluded 0 rows from the RAG corpus. Either the golden set "
            "shares no source rows with the corpus (possible, but check) or the id "
            "formats have drifted apart again -- see _id_variants()."
        )
    else:
        logger.info(f"Leakage guard excluded {skipped} golden-set source rows from the RAG corpus.")
    return records


class HistoricalVectorStore:
    """Manages the embedded ChromaDB vector collection for customer support RAG."""

    def __init__(self, persist_dir: str | None = None, collection_name: str = "apple_support_resolutions"):
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
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
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
            logger.info(
                f"Initializing vector store with {len(real_records)} real "
                f"@AppleSupport pairs (golden-set examples excluded)..."
            )
            self.index_records(real_records)
        else:
            logger.warning(
                "No real Kaggle pairs found at KAGGLE_PAIRS_PATH -- "
                "falling back to the small hand-written seed corpus. "
                "Run `python -m src.data.ingest_kaggle extract` first "
                "for real grounding."
            )
            self.index_records(HISTORICAL_APPLE_RESOLUTIONS)

    def index_records(self, records: list[dict[str, str]]):
        """Indexes a list of resolution records into ChromaDB."""
        ids = []
        documents = []
        metadatas = []
        embeddings = []

        texts_to_embed = [r["customer_text"] for r in records]
        encoded_vecs = self.encoder.encode(texts_to_embed, convert_to_numpy=True, normalize_embeddings=True)

        for record, vec in zip(records, encoded_vecs, strict=False):
            ids.append(record["tweet_id"])
            documents.append(record["customer_text"])
            metadatas.append(
                {
                    "agent_reply": record["agent_reply"],
                    "intent": record.get("intent", "OS_SOFTWARE_TROUBLESHOOTING"),
                }
            )
            embeddings.append(vec.tolist())

        # chromadb's Mapping-based metadata type is narrower than the dicts it
        # actually accepts at runtime; this call is exercised by the test suite.
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,  # type: ignore[arg-type]
            embeddings=embeddings,
        )

    def query(self, query_text: str, intent: str | None = None, top_k: int = 3) -> dict:
        """Queries the collection for semantically similar historical customer tweets."""
        query_vec = self.encoder.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()

        where_filter = {"intent": intent} if intent and intent != "OUT_OF_SCOPE_AMBIGUOUS" else None

        # chromadb's stubs declare narrower Mapping/Where types than the plain
        # dicts it accepts and returns at runtime; both calls are exercised by
        # the test suite and by every eval run.
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            where=where_filter,  # type: ignore[arg-type]
            include=["documents", "metadatas", "distances"],
        )
        return results  # type: ignore[return-value]
