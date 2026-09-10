"""Baseline intent classifiers for benchmark comparison (Deliverable 4)."""

from typing import Dict, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.models import AppleIntentEnum, IntentResult
from src.intent.taxonomy import INTENT_PROTOTYPES


class TrivialMajorityClassifier:
    """Baseline 1 (Trivial): Always predicts the majority class."""

    def __init__(self, majority_class: AppleIntentEnum = AppleIntentEnum.OS_SOFTWARE_TROUBLESHOOTING):
        self.majority_class = majority_class

    def predict(self, text: str) -> IntentResult:
        dist = {intent.value: (1.0 if intent == self.majority_class else 0.0) for intent in AppleIntentEnum}
        return IntentResult(
            primary_intent=self.majority_class,
            confidence=0.50,  # Fixed naive baseline confidence
            secondary_intents=[],
            score_distribution=dist,
        )


class SimpleTfidfClassifier:
    """Baseline 2 (Simple): TF-IDF n-grams + Logistic Regression."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1500)
        self.model = LogisticRegression(max_iter=500, random_state=42)
        self.classes: List[str] = [e.value for e in AppleIntentEnum]
        self._is_fitted = False
        self._fit_default_corpus()

    def _fit_default_corpus(self):
        """Fits on canonical prototype samples to initialize the simple baseline."""
        texts = []
        labels = []
        for intent, prototypes in INTENT_PROTOTYPES.items():
            for p in prototypes:
                texts.append(p)
                labels.append(intent.value)

        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self._is_fitted = True

    def fit(self, texts: List[str], labels: List[str]):
        """Fits on custom labeled dataset."""
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.classes = list(self.model.classes_)
        self._is_fitted = True

    def predict(self, text: str) -> IntentResult:
        if not self._is_fitted:
            self._fit_default_corpus()

        X = self.vectorizer.transform([text])
        probs = self.model.predict_proba(X)[0]
        class_probs = {cls_name: float(p) for cls_name, p in zip(self.model.classes_, probs)}

        # Sort by probability
        sorted_intents = sorted(class_probs.items(), key=lambda item: item[1], reverse=True)
        top_intent_str, top_conf = sorted_intents[0]
        second_intent_str, second_conf = sorted_intents[1]

        secondary = []
        if (top_conf - second_conf) < 0.15:
            secondary.append(AppleIntentEnum(second_intent_str))

        return IntentResult(
            primary_intent=AppleIntentEnum(top_intent_str),
            confidence=round(top_conf, 4),
            secondary_intents=secondary,
            score_distribution=class_probs,
        )
