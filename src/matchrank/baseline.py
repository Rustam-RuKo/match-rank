from __future__ import annotations

from matchrank.domain import Applicant, University
from matchrank.features import pair_features


def baseline_score(applicant: Applicant, university: University) -> float:
    """Transparent weighted-filter baseline on a roughly 0-1 scale."""
    f = pair_features(applicant, university)
    admission = 1.0 if f["grade_margin"] >= 0 else max(0.0, 1.0 + f["grade_margin"])
    affordability = min(1.0, f["effective_cost_ratio"])
    return (
        0.28 * f["major_match"]
        + 0.18 * affordability
        + 0.16 * admission
        + 0.12 * f["location_match"]
        + 0.10 * f["language_match"]
        + 0.08 * f["scholarship_match"]
        + 0.05 * f["prestige_score"]
        + 0.03 * f["acceptance_rate"]
    )
