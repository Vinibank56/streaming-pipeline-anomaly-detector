"""Objective invariants for anomaly detection results."""

from __future__ import annotations

from detector.types import AnomalyResult


def assert_result_invariants(result: AnomalyResult, *, threshold: float) -> None:
    assert isinstance(result.is_anomaly, bool)
    assert isinstance(result.z_score, float)
    assert isinstance(result.baseline_mean, float)
    assert isinstance(result.baseline_std, float)
    assert result.baseline_std >= 0
    assert isinstance(result.consecutive_high, int)
    assert result.consecutive_high >= 0
    assert isinstance(result.warmed_up, bool)

    audit = result.audit
    assert set(audit.keys()) == {
        "stream_id",
        "sample_count",
        "ema_alpha_applied",
        "threshold",
        "warmed_up",
        "consecutive_required",
    }
    assert audit["threshold"] == threshold

    if not result.warmed_up:
        assert result.is_anomaly is False

    if result.is_anomaly:
        assert result.warmed_up is True
        assert result.consecutive_high >= audit["consecutive_required"]

    if not result.is_anomaly and result.consecutive_high == 0:
        assert abs(result.z_score) <= threshold or not result.warmed_up
