from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from matchrank.data import generate_catalog, load_catalog
from matchrank.domain import Applicant
from matchrank.features import pair_features
from matchrank.monitoring import PredictionMonitor
from matchrank.ranker import MatchRanker


class ApplicantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    applicant_id: str = Field(min_length=1, max_length=100)
    grade: float = Field(ge=0, le=100)
    budget_usd: float = Field(gt=0, le=1_000_000)
    preferred_majors: list[str] = Field(default_factory=list, max_length=10)
    preferred_countries: list[str] = Field(default_factory=list, max_length=10)
    languages: list[str] = Field(default_factory=list, max_length=10)
    needs_scholarship: bool = False
    test_score: float | None = Field(default=None, ge=0, le=2400)
    top_k: int = Field(default=10, ge=1, le=50)

    def domain(self) -> Applicant:
        data = self.model_dump(exclude={"top_k"})
        for name in ("preferred_majors", "preferred_countries", "languages"):
            data[name] = tuple(data[name])
        return Applicant(**data)


class Recommendation(BaseModel):
    rank: int
    university_id: str
    university_name: str
    score: float
    source: str
    explanations: list[str]


class RecommendationResponse(BaseModel):
    model_version: str
    applicant_id: str
    recommendations: list[Recommendation]


def create_app(artifact_dir: Path | None = None) -> FastAPI:
    artifact_dir = artifact_dir or Path(os.getenv("MATCHRANK_ARTIFACT_DIR", "artifacts"))
    model_path, catalog_path = artifact_dir / "model.joblib", artifact_dir / "universities.json"
    ranker = MatchRanker.load(str(model_path)) if model_path.exists() else MatchRanker()
    universities = load_catalog(catalog_path) if catalog_path.exists() else generate_catalog()
    monitor = PredictionMonitor()
    app = FastAPI(title="MatchRank", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "model_loaded": ranker.fitted, "catalog_size": len(universities)}

    @app.post("/v1/recommendations", response_model=RecommendationResponse)
    def recommend(payload: ApplicantRequest) -> RecommendationResponse:
        if payload.top_k > len(universities):
            raise HTTPException(422, "top_k exceeds catalog size")
        applicant = payload.domain()
        with monitor.timer():
            ranked = ranker.rank(applicant, universities, payload.top_k)
            for university in universities:
                monitor.observe_features(pair_features(applicant, university))
        return RecommendationResponse(
            model_version="0.1.0",
            applicant_id=applicant.applicant_id,
            recommendations=[
                Recommendation(
                    rank=index,
                    university_id=item.university.university_id,
                    university_name=item.university.name,
                    score=round(item.score, 6),
                    source=item.source,
                    explanations=list(item.explanations),
                )
                for index, item in enumerate(ranked, 1)
            ],
        )

    @app.get("/monitoring/drift")
    def drift() -> dict[str, object]:
        return monitor.drift(ranker.reference_samples)

    @app.get("/metrics")
    def metrics() -> Response:
        latency = monitor.latency_summary()
        body = "\n".join(
            [
                "# HELP matchrank_prediction_requests_total Number of recommendation requests.",
                "# TYPE matchrank_prediction_requests_total counter",
                f"matchrank_prediction_requests_total {latency['requests']}",
                "# HELP matchrank_prediction_latency_ms Prediction latency in milliseconds.",
                "# TYPE matchrank_prediction_latency_ms gauge",
                f"matchrank_prediction_latency_ms{{quantile=\"mean\"}} {latency['mean_ms']}",
                f"matchrank_prediction_latency_ms{{quantile=\"0.95\"}} {latency['p95_ms']}",
            ]
        )
        return Response(body + "\n", media_type="text/plain; version=0.0.4")

    return app


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run("matchrank.api:app", host="0.0.0.0", port=8000)
