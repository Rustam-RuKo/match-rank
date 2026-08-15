from fastapi.testclient import TestClient

from matchrank.api import create_app


def test_recommendation_contract(tmp_path) -> None:
    client = TestClient(create_app(tmp_path))
    response = client.post(
        "/v1/recommendations",
        json={
            "applicant_id": "candidate-1",
            "grade": 88,
            "budget_usd": 35000,
            "preferred_majors": ["computer science"],
            "preferred_countries": ["Canada"],
            "languages": ["English"],
            "needs_scholarship": True,
            "test_score": 1350,
            "top_k": 3,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["recommendations"]) == 3
    assert body["recommendations"][0]["rank"] == 1
    assert body["recommendations"][0]["explanations"]
    assert client.get("/metrics").status_code == 200
