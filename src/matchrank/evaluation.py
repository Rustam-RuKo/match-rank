from __future__ import annotations

from collections.abc import Callable, Iterable
from math import log2

import numpy as np

from matchrank.domain import Applicant, University

Scorer = Callable[[Applicant, University], float]


def precision_at_k(relevances: list[float], k: int, threshold: float = 2.0) -> float:
    chosen = relevances[:k]
    return sum(value >= threshold for value in chosen) / k if k else 0.0


def recall_at_k(
    relevances: list[float], all_relevances: list[float], k: int, threshold: float = 2.0
) -> float:
    relevant_total = sum(value >= threshold for value in all_relevances)
    return (
        sum(value >= threshold for value in relevances[:k]) / relevant_total
        if relevant_total
        else 0.0
    )


def ndcg_at_k(relevances: list[float], k: int) -> float:
    def dcg(values: list[float]) -> float:
        return sum((2**value - 1) / log2(index + 2) for index, value in enumerate(values))

    actual = dcg(relevances[:k])
    ideal = dcg(sorted(relevances, reverse=True)[:k])
    return actual / ideal if ideal else 0.0


def evaluate(
    applicants: Iterable[Applicant],
    universities: list[University],
    relevance: dict[tuple[str, str], float],
    scorer: Scorer,
    k: int = 10,
) -> dict[str, float]:
    rows = []
    for applicant in applicants:
        ordered = sorted(universities, key=lambda u: scorer(applicant, u), reverse=True)
        ranked = [relevance[(applicant.applicant_id, u.university_id)] for u in ordered]
        all_values = [relevance[(applicant.applicant_id, u.university_id)] for u in universities]
        rows.append(
            (precision_at_k(ranked, k), recall_at_k(ranked, all_values, k), ndcg_at_k(ranked, k))
        )
    values = np.asarray(rows)
    return {
        f"precision@{k}": float(values[:, 0].mean()),
        f"recall@{k}": float(values[:, 1].mean()),
        f"ndcg@{k}": float(values[:, 2].mean()),
        "profiles": int(len(rows)),
    }
