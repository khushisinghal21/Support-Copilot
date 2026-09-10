"""Production Semantic Centroid Intent Classifier with calibrated confidence scoring."""

from typing import Dict, List, Optional
import numpy as np
from src.models import AppleIntentEnum, IntentResult
from src.intent.taxonomy import INTENT_PROTOTYPES
from src.config import EMBEDDING_MODEL_NAME, MIN_INTENT_CONFIDENCE


class SemanticCentroidClassifier:
    """Production Intent Classifier using dense semantic centroids and calibrated scoring."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        confidence_threshold: float = MIN_INTENT_CONFIDENCE,
        temperature: float = 0.08,
    ):
        # get_encoder() is imported here, not at module level -- see the
        # matching note in src/drafting/vector_store.py for why: a
        # module-level import would make every "import src.server" pay
        # sentence_transformers/torch's import cost, which is slow enough on
        # a low-CPU deploy host to block the ASGI server's port from opening
        # before the platform's port-scan timeout.
        #
        # get_encoder() (src/embeddings.py) also caches the model by name,
        # so this doesn't load a second copy of the encoder into memory when
        # HistoricalVectorStore has already loaded the same model -- two
        # independent SentenceTransformer(EMBEDDING_MODEL_NAME) instances was
        # a contributor to an out-of-memory crash on Render's 512MB tier.
        from src.embeddings import get_encoder

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.temperature = temperature
        self.encoder = get_encoder(model_name)
        self.centroids: Dict[AppleIntentEnum, np.ndarray] = {}
        self._build_default_centroids()

    def _build_default_centroids(self):
        """Computes average embedding vector for each canonical intent prototype."""
        for intent, prototypes in INTENT_PROTOTYPES.items():
            embeddings = self.encoder.encode(prototypes, convert_to_numpy=True, normalize_embeddings=True)
            centroid = np.mean(embeddings, axis=0)
            centroid = centroid / np.linalg.norm(centroid)  # L2 normalize
            self.centroids[intent] = centroid

    def predict(self, text: str) -> IntentResult:
        """Classifies a tweet into an AppleIntentEnum with calibrated confidence."""
        if not text or not text.strip():
            return IntentResult(
                primary_intent=AppleIntentEnum.OUT_OF_SCOPE_AMBIGUOUS,
                confidence=0.0,
                secondary_intents=[],
                score_distribution={},
            )

        # Encode query
        query_vec = self.encoder.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]

        # Calculate cosine similarity with each intent centroid
        similarities: Dict[AppleIntentEnum, float] = {}
        for intent, centroid in self.centroids.items():
            cos_sim = float(np.dot(query_vec, centroid))
            similarities[intent] = cos_sim

        # Softmax temperature scaling over cosine similarities for well-calibrated probabilities
        intents_list = list(similarities.keys())
        sim_values = np.array([similarities[i] for i in intents_list])
        exp_scaled = np.exp((sim_values - np.max(sim_values)) / self.temperature)
        probs = exp_scaled / np.sum(exp_scaled)

        prob_dist = {i.value: float(p) for i, p in zip(intents_list, probs)}

        # Rank by probability
        ranked_indices = np.argsort(probs)[::-1]
        top_intent = intents_list[ranked_indices[0]]
        top_prob = float(probs[ranked_indices[0]])
        second_intent = intents_list[ranked_indices[1]]
        second_prob = float(probs[ranked_indices[1]])

        secondary_intents = []
        if (top_prob - second_prob) < 0.15 and second_intent != AppleIntentEnum.OUT_OF_SCOPE_AMBIGUOUS:
            secondary_intents.append(second_intent)

        # Fallback to OUT_OF_SCOPE_AMBIGUOUS if below confidence threshold
        if top_prob < self.confidence_threshold:
            return IntentResult(
                primary_intent=AppleIntentEnum.OUT_OF_SCOPE_AMBIGUOUS,
                confidence=round(top_prob, 4),
                secondary_intents=[top_intent] if top_intent != AppleIntentEnum.OUT_OF_SCOPE_AMBIGUOUS else [],
                score_distribution=prob_dist,
            )

        return IntentResult(
            primary_intent=top_intent,
            confidence=round(top_prob, 4),
            secondary_intents=secondary_intents,
            score_distribution=prob_dist,
        )
