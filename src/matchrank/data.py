from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from matchrank.domain import Applicant, University
from matchrank.features import latent_relevance, pair_features

MAJORS = ("computer science", "business", "engineering", "medicine", "economics", "design")
COUNTRIES = ("United States", "Canada", "United Kingdom", "Germany", "Australia")
LANGUAGES = ("English", "German")


def generate_catalog(size: int = 60, seed: int = 42) -> list[University]:
    rng = np.random.default_rng(seed)
    universities = []
    for index in range(size):
        country = COUNTRIES[index % len(COUNTRIES)]
        languages = ("German", "English") if country == "Germany" else ("English",)
        universities.append(
            University(
                university_id=f"uni-{index:03d}",
                name=f"{country} Institute {index + 1}",
                country=country,
                tuition_usd=float(rng.integers(8_000, 65_000)),
                majors=tuple(rng.choice(MAJORS, size=2, replace=False).tolist()),
                languages=languages,
                min_grade=float(rng.integers(65, 94)),
                min_test_score=float(rng.integers(900, 1450)),
                scholarship_available=bool(rng.random() < 0.55),
                scholarship_rate=float(rng.uniform(0.15, 0.8)),
                acceptance_rate=float(rng.uniform(0.08, 0.85)),
                prestige_score=float(rng.uniform(0.2, 0.98)),
            )
        )
    return universities


def generate_applicants(size: int = 300, seed: int = 43) -> list[Applicant]:
    rng = np.random.default_rng(seed)
    return [
        Applicant(
            applicant_id=f"app-{index:04d}",
            grade=float(rng.integers(62, 100)),
            budget_usd=float(rng.integers(10_000, 70_000)),
            preferred_majors=(str(rng.choice(MAJORS)),),
            preferred_countries=tuple(rng.choice(COUNTRIES, size=rng.integers(1, 3), replace=False)),
            languages=("English",) if rng.random() < 0.85 else ("German",),
            needs_scholarship=bool(rng.random() < 0.5),
            test_score=float(rng.integers(850, 1600)),
        )
        for index in range(size)
    ]

def generate_labels(
    applicants: list[Applicant], universities: list[University], seed: int = 44
) -> dict[tuple[str, str], float]:
    rng = np.random.default_rng(seed)
    return {
        (applicant.applicant_id, university.university_id): latent_relevance(
            pair_features(applicant, university), float(rng.normal(0, 0.18))
        )
        for applicant in applicants
        for university in universities
    }


def save_catalog(universities: list[University], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([u.to_dict() for u in universities], indent=2), encoding="utf-8")


def load_catalog(path: Path) -> list[University]:
    return [
        University(
            **{
                **row,
                "majors": tuple(row["majors"]),
                "languages": tuple(row["languages"]),
            }
        )
        for row in json.loads(path.read_text(encoding="utf-8"))
    ]
