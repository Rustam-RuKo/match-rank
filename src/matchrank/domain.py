from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Applicant:
    applicant_id: str
    grade: float
    budget_usd: float
    preferred_majors: tuple[str, ...] = ()
    preferred_countries: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    needs_scholarship: bool = False
    test_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class University:
    university_id: str
    name: str
    country: str
    tuition_usd: float
    majors: tuple[str, ...]
    languages: tuple[str, ...]
    min_grade: float
    min_test_score: float | None = None
    scholarship_available: bool = False
    scholarship_rate: float = 0.0
    acceptance_rate: float = 0.5
    prestige_score: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
