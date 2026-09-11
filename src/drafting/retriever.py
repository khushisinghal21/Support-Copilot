"""Semantic retriever for grounding replies on historical resolutions."""

from src.drafting.vector_store import HistoricalVectorStore
from src.models import HistoricalCitation, RetrievalResult


class HistoricalRetriever:
    """Retrieves top-k historical Apple Support resolution pairs."""

    def __init__(self, vector_store: HistoricalVectorStore | None = None):
        self.vector_store = vector_store or HistoricalVectorStore()

    def retrieve(self, query: str, intent: str | None = None, k: int = 3) -> RetrievalResult:
        """Retrieves top-k relevant resolution snippets and returns structured RetrievalResult."""
        results = self.vector_store.query(query_text=query, intent=intent, top_k=k)

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        citations: list[HistoricalCitation] = []
        snippets: list[str] = []
        scores: list[float] = []

        for doc_id, doc_text, meta, dist in zip(ids, documents, metadatas, distances, strict=False):
            # Chroma cosine distance = 1 - cosine_similarity
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))
            agent_reply = meta.get("agent_reply", "")

            citation = HistoricalCitation(
                tweet_id=doc_id,
                customer_text=doc_text,
                agent_reply=agent_reply,
                similarity_score=round(similarity, 4),
            )
            citations.append(citation)
            snippets.append(agent_reply)
            scores.append(round(similarity, 4))

        max_sim = max(scores) if scores else 0.0

        return RetrievalResult(
            snippets=snippets,
            citations=citations,
            similarity_scores=scores,
            max_similarity=round(max_sim, 4),
        )
