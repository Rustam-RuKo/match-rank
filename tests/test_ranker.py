from matchrank.data import generate_applicants, generate_catalog, generate_labels
from matchrank.ranker import MatchRanker


def test_ranker_trains_and_returns_explanations() -> None:
    universities = generate_catalog(12)
    applicants = generate_applicants(20)
    labels = generate_labels(applicants, universities)
    examples = [
        (a, u, labels[(a.applicant_id, u.university_id)])
        for a in applicants[:15]
        for u in universities
    ]
    ranker = MatchRanker().fit(examples)
    results = ranker.rank(applicants[-1], universities, top_k=5)
    assert len(results) == 5
    assert all(item.source == "learning_to_rank" for item in results)
    assert all(item.explanations for item in results)
    assert [item.score for item in results] == sorted(
        [item.score for item in results], reverse=True
    )


def test_incomplete_profile_uses_cold_start() -> None:
    university = generate_catalog(1)
    applicant = generate_applicants(1)[0]
    incomplete = type(applicant)(
        applicant_id="new", grade=applicant.grade, budget_usd=applicant.budget_usd
    )
    result = MatchRanker().rank(incomplete, university, 1)[0]
    assert result.source == "cold_start_baseline"
