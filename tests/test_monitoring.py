from matchrank.monitoring import PredictionMonitor, population_stability_index


def test_psi_detects_shift() -> None:
    reference = [value / 100 for value in range(100)]
    shifted = [5 + value / 100 for value in range(100)]
    assert population_stability_index(reference, reference) == 0.0
    assert population_stability_index(reference, shifted) > 0.2


def test_monitor_reports_feature_drift() -> None:
    monitor = PredictionMonitor()
    for value in range(20):
        monitor.observe_features({"grade_margin": 5.0 + value / 100})
    result = monitor.drift({"grade_margin": [value / 100 for value in range(100)]})
    assert result["status"] == "drift_detected"
