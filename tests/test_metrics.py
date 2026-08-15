from matchrank.evaluation import ndcg_at_k, precision_at_k, recall_at_k


def test_ranking_metrics() -> None:
    ranked = [3.0, 0.0, 2.0, 1.0]
    assert precision_at_k(ranked, 2) == 0.5
    assert recall_at_k(ranked, ranked, 3) == 1.0
    assert 0.8 < ndcg_at_k(ranked, 4) < 1.0


def test_perfect_ndcg() -> None:
    assert ndcg_at_k([3.0, 2.0, 1.0], 3) == 1.0
