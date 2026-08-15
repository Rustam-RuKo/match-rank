from __future__ import annotations

import argparse
import json
from pathlib import Path

from matchrank.baseline import baseline_score
from matchrank.data import generate_applicants, generate_catalog, generate_labels, save_catalog
from matchrank.evaluation import evaluate
from matchrank.ranker import MatchRanker


def train(output_dir: Path, profiles: int = 300, k: int = 10) -> dict[str, object]:
    universities = generate_catalog()
    applicants = generate_applicants(profiles)
    labels = generate_labels(applicants, universities)
    split = int(profiles * 0.8)
    train_profiles, test_profiles = applicants[:split], applicants[split:]
    examples = [
        (applicant, university, labels[(applicant.applicant_id, university.university_id)])
        for applicant in train_profiles
        for university in universities
    ]
    ranker = MatchRanker().fit(examples)
    metrics = {
        "baseline": evaluate(test_profiles, universities, labels, baseline_score, k),
        "model": evaluate(test_profiles, universities, labels, ranker.predict_score, k),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    ranker.save(str(output_dir / "model.joblib"))
    save_catalog(universities, output_dir / "universities.json")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _log_mlflow(metrics)
    return metrics


def _log_mlflow(metrics: dict[str, object]) -> None:
    try:
        import mlflow
    except ImportError:
        return
    with mlflow.start_run(run_name="matchrank-training"):
        for family, values in metrics.items():
            for name, value in values.items():
                if name != "profiles":
                    mlflow.log_metric(f"{family}_{name}", value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--profiles", type=int, default=300)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(train(args.output_dir, args.profiles, args.k), indent=2))


if __name__ == "__main__":
    main()
