from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from matchrank.baseline import baseline_score
from matchrank.domain import Applicant, University
from matchrank.features import NUMERIC_FEATURES, pair_features, profile_completeness


@dataclass(frozen=True)
class RankedUniversity:
    university: University
    score: float
    source: str
    explanations: tuple[str, ...]


class MatchRanker:
    """Pointwise learning-to-rank model; candidates are sorted by predicted relevance."""

    def __init__(self, random_state: int = 42) -> None:
        self.model = HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=180,
            max_leaf_nodes=15,
            l2_regularization=0.2,
            random_state=random_state,
        )
        self.feature_names = list(NUMERIC_FEATURES)
        self.reference_means: dict[str, float] = {}
        self.reference_samples: dict[str, list[float]] = {}
        self.fitted = False

    def _matrix(self, rows: Iterable[dict[str, float]]) -> pd.DataFrame:
        return pd.DataFrame.from_records(rows, columns=self.feature_names, coerce_float=True)

    def fit(
        self,
        examples: Iterable[tuple[Applicant, University, float]],
    ) -> "MatchRanker":
        materialized = list(examples)
        rows = [pair_features(a, u) for a, u, _ in materialized]
        labels = np.asarray([label for _, _, label in materialized])
        matrix = self._matrix(rows)
        self.model.fit(matrix, labels)
        self.reference_means = {
            name: float(matrix[name].mean()) for name in self.feature_names
        }
        self.reference_samples = {
            name: [
                float(value)
                for value in matrix[name].iloc[:: max(1, len(matrix) // 500)]
            ]
            for name in self.feature_names
        }
        self.fitted = True
        return self

    def predict_score(self, applicant: Applicant, university: University) -> float:
        if not self.fitted:
            return baseline_score(applicant, university)
        prediction = self.model.predict(self._matrix([pair_features(applicant, university)]))[0]
        return float(np.clip(prediction, 0.0, 4.0))

    def rank(
        self,
        applicant: Applicant,
        universities: Iterable[University],
        top_k: int = 10,
    ) -> list[RankedUniversity]:
        candidates = list(universities)
        cold_start = not self.fitted or profile_completeness(applicant) < 0.5
        source = "cold_start_baseline" if cold_start else "learning_to_rank"
        if cold_start:
            scores = [baseline_score(applicant, university) for university in candidates]
        else:
            rows = [pair_features(applicant, university) for university in candidates]
            scores = np.clip(self.model.predict(self._matrix(rows)), 0.0, 4.0).tolist()
        scored = list(zip(candidates, scores, strict=True))
        scored.sort(key=lambda item: (-item[1], item[0].university_id))
        return [
            RankedUniversity(u, score, source, self.explain(applicant, u, cold_start))
            for u, score in scored[:top_k]
        ]

    def explain(
        self, applicant: Applicant, university: University, cold_start: bool = False
    ) -> tuple[str, ...]:
        f = pair_features(applicant, university)
        labels = {
            "major_match": "preferred major is offered",
            "location_match": "location matches your preference",
            "language_match": "teaching language matches",
            "scholarship_match": "scholarship needs are supported",
            "effective_cost_ratio": "estimated cost fits your budget",
            "grade_margin": "grades meet the admission threshold",
            "prestige_score": "strong university reputation",
            "acceptance_rate": "comparatively accessible admission rate",
        }
        if cold_start or not self.fitted:
            priority = [
                "major_match", "effective_cost_ratio", "grade_margin", "location_match",
                "language_match", "scholarship_match", "prestige_score",
            ]
            return tuple(labels[name] for name in priority if f[name] >= 0.5)[:3]

        changed_rows = []
        for name, label in labels.items():
            changed = dict(f)
            changed[name] = self.reference_means[name]
            changed_rows.append(changed)
        predictions = np.clip(self.model.predict(self._matrix([f, *changed_rows])), 0.0, 4.0)
        original = float(predictions[0])
        impacts = [
            (original - float(prediction), label)
            for prediction, label in zip(predictions[1:], labels.values(), strict=True)
        ]
        positive = [label for impact, label in sorted(impacts, reverse=True) if impact > 0.005]
        return tuple(positive[:3] or ["overall profile fit is competitive"])

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "MatchRanker":
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError("artifact does not contain a MatchRanker")
        return loaded
