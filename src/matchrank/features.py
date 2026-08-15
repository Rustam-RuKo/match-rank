from __future__ import annotations

from math import exp

from matchrank.domain import Applicant, University

NUMERIC_FEATURES = (
    "grade_margin",
    "test_margin",
    "budget_ratio",
    "effective_cost_ratio",
    "major_match",
    "location_match",
    "language_match",
    "scholarship_match",
    "acceptance_rate",
    "prestige_score",
)


def _overlap(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if not left:
        return 0.5  # neutral for an unspecified preference
    return float(bool({x.casefold() for x in left} & {x.casefold() for x in right}))


def pair_features(applicant: Applicant, university: University) -> dict[str, float]:
    effective_cost = university.tuition_usd
    if applicant.needs_scholarship and university.scholarship_available:
        effective_cost *= 1.0 - university.scholarship_rate
    test_margin = 0.0
    if university.min_test_score is not None:
        test_margin = (
            -0.5
            if applicant.test_score is None
            else (applicant.test_score - university.min_test_score) / 100.0
        )
    return {
        "grade_margin": (applicant.grade - university.min_grade) / 10.0,
        "test_margin": test_margin,
        "budget_ratio": applicant.budget_usd / max(university.tuition_usd, 1.0),
        "effective_cost_ratio": applicant.budget_usd / max(effective_cost, 1.0),
        "major_match": _overlap(applicant.preferred_majors, university.majors),
        "location_match": _overlap(applicant.preferred_countries, (university.country,)),
        "language_match": _overlap(applicant.languages, university.languages),
        "scholarship_match": float(
            not applicant.needs_scholarship or university.scholarship_available
        ),
        "acceptance_rate": university.acceptance_rate,
        "prestige_score": university.prestige_score,
    }


def profile_completeness(applicant: Applicant) -> float:
    optional = (
        bool(applicant.preferred_majors),
        bool(applicant.preferred_countries),
        bool(applicant.languages),
        applicant.test_score is not None,
    )
    return sum(optional) / len(optional)


def latent_relevance(features: dict[str, float], noise: float = 0.0) -> float:
    """Synthetic ground truth used only to make the demo data reproducible."""
    affordability = min(features["effective_cost_ratio"], 1.5)
    admission = 1.0 / (1.0 + exp(-5.0 * features["grade_margin"]))
    raw = (
        1.8 * features["major_match"]
        + 1.1 * features["location_match"]
        + 0.8 * features["language_match"]
        + 1.3 * affordability
        + 1.1 * admission
        + 0.7 * features["scholarship_match"]
        + 0.5 * features["prestige_score"]
        + 0.2 * features["acceptance_rate"]
        + noise
    )
    return max(0.0, min(4.0, raw / 1.8))
