from __future__ import annotations

from collections import deque
from threading import Lock
from time import perf_counter

import numpy as np


def population_stability_index(reference: list[float], current: list[float]) -> float:
    if len(reference) < 2 or len(current) < 2:
        return 0.0
    boundaries = np.unique(np.quantile(reference, np.linspace(0, 1, 11)))
    if len(boundaries) < 3:
        return 0.0
    boundaries[0], boundaries[-1] = -np.inf, np.inf
    expected, _ = np.histogram(reference, bins=boundaries)
    observed, _ = np.histogram(current, bins=boundaries)
    expected_rate = np.maximum(expected / expected.sum(), 1e-6)
    observed_rate = np.maximum(observed / observed.sum(), 1e-6)
    return float(np.sum((observed_rate - expected_rate) * np.log(observed_rate / expected_rate)))


class PredictionMonitor:
    def __init__(self, max_samples: int = 2_000) -> None:
        self.latencies_ms: deque[float] = deque(maxlen=max_samples)
        self.feature_samples: dict[str, deque[float]] = {}
        self.requests = 0
        self._lock = Lock()

    def timer(self):
        monitor = self

        class Timer:
            def __enter__(self):
                self.started = perf_counter()
                return self

            def __exit__(self, *_):
                with monitor._lock:
                    monitor.requests += 1
                    monitor.latencies_ms.append((perf_counter() - self.started) * 1000)

        return Timer()

    def observe_features(self, features: dict[str, float]) -> None:
        with self._lock:
            for name, value in features.items():
                self.feature_samples.setdefault(name, deque(maxlen=2_000)).append(value)

    def latency_summary(self) -> dict[str, float | int]:
        values = list(self.latencies_ms)
        return {
            "requests": self.requests,
            "mean_ms": float(np.mean(values)) if values else 0.0,
            "p95_ms": float(np.percentile(values, 95)) if values else 0.0,
        }

    def drift(self, reference: dict[str, list[float]]) -> dict[str, object]:
        scores = {
            name: population_stability_index(values, list(self.feature_samples.get(name, ())))
            for name, values in reference.items()
            if len(self.feature_samples.get(name, ())) >= 10
        }
        return {
            "status": "drift_detected" if any(value >= 0.2 for value in scores.values()) else "ok",
            "threshold": 0.2,
            "psi": scores,
            "samples_required": 10,
        }
