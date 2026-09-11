"""Shared SentenceTransformer encoder factory.

SemanticCentroidClassifier (src/intent/classifier.py) and
HistoricalVectorStore (src/drafting/vector_store.py) both need a text
encoder, and both were independently constructing their own
SentenceTransformer(EMBEDDING_MODEL_NAME) instance -- same model
(all-MiniLM-L6-v2), loaded into memory twice for no reason, whenever a
SupportPipeline() builds both together. On a memory-constrained host (e.g.
Render's free tier, 512MB) that duplication was one of the contributors to
an out-of-memory crash/restart in production.

get_encoder() caches one instance per model_name (in practice, always the
same one) so both callers share the same underlying model weights.

Import is function-local here too, not at module level -- for the same
reason it's function-local in classifier.py/vector_store.py: importing
sentence_transformers (and the torch it pulls in) is itself slow enough on
a low-CPU host to risk blocking the ASGI server's port from opening before
the platform's port-scan timeout. Keeping the import inside get_encoder()
means "import src.embeddings" alone stays cheap.
"""

_ENCODER_CACHE: dict[str, object] = {}


def get_encoder(model_name: str):
    """Returns a shared SentenceTransformer instance for model_name, creating
    and caching it on first call. Safe to call from multiple classes/places --
    the actual model is only ever loaded into memory once per model_name."""
    if model_name not in _ENCODER_CACHE:
        from sentence_transformers import SentenceTransformer

        _ENCODER_CACHE[model_name] = SentenceTransformer(model_name)
    return _ENCODER_CACHE[model_name]
