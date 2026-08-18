"""Hidden statistical and ML-behavior tests — sealed from the agent."""

from __future__ import annotations

import math

import pytest

from detector import ema
from helpers import default_config, feed, load_detector_module
from invariants import assert_result_invariants
from spec_reference import ReferenceStreamingAnomalyDetector


@pytest.fixture(scope="module")
def detector_cls():
    return load_detector_module().StreamingAnomalyDetector


def test_must_detect_sustained_attack_on_unpublished_stream(detector_cls):
    """Sensitivity: sustained spikes after warmup must eventually alert."""
    from detector.config import DetectorConfig

    cfg = DetectorConfig(
        ema_alpha=0.05,
        z_threshold=2.5,
        warmup_samples=5,
        consecutive_required=2,
        min_std=0.1,
    )
    results = feed(detector_cls(cfg), "wh_stat_attack", [10.0] * 5 + [80.0, 80.0])
    assert results[-1].is_anomaly is True
    assert results[-1].consecutive_high >= cfg.consecutive_required


def test_benign_gaussian_sequence_false_positive_rate_is_zero(detector_cls):
    """ML metric: FP rate must be 0 on stable noisy baseline (n=50)."""
    import random

    cfg = default_config()
    detector = detector_cls(cfg)
    random.seed(991737)
    values = [42.0 + random.gauss(0, 0.08) for _ in range(50)]
    results = feed(detector, "wh_stat_benign50", values)
    fp_count = sum(r.is_anomaly for r in results)
    assert fp_count == 0


def test_z_score_matches_ema_helper_pre_update(detector_cls):
    """Verify statistical model uses ema.z_score on pre-update baseline."""
    from detector.config import DetectorConfig
    from detector.types import MetricPoint

    cfg = DetectorConfig(0.15, 3.0, 4, 2, 0.1)
    detector = detector_cls(cfg)
    history = [5.0, 5.0, 5.0, 5.0, 7.5]
    for index, value in enumerate(history):
        detector.update(MetricPoint("wh_stat_z", f"2026-07-01T00:00:{index:02d}Z", value))

    probe = MetricPoint("wh_stat_z", "2026-07-01T00:00:10Z", 9.0)
    state_before = detector._state("wh_stat_z")  # noqa: SLF001 — sealed test
    expected_z = ema.z_score(
        probe.value,
        state_before.mean,
        ema.effective_std(state_before.var, cfg.min_std),
    )
    result = detector.update(probe)
    assert result.z_score == pytest.approx(expected_z, rel=1e-9, abs=1e-9)


def test_ema_alpha_changes_baseline_trajectory(detector_cls):
    """Anti rule-based: ignoring ema_alpha must fail this statistical check."""
    from detector.config import DetectorConfig

    values = [10.0] * 5 + [30.0] * 6
    slow = feed(
        detector_cls(DetectorConfig(0.05, 3.0, 5, 2, 0.1)),
        "wh_stat_slow",
        values,
    )[-1]
    fast = feed(
        detector_cls(DetectorConfig(0.85, 3.0, 5, 2, 0.1)),
        "wh_stat_fast",
        values,
    )[-1]
    assert slow.baseline_mean != pytest.approx(fast.baseline_mean, abs=1e-6)
    assert fast.baseline_mean > slow.baseline_mean


def test_concept_drift_adaptation_reduces_z_after_regime_shift(detector_cls):
    """After regime shift, high alpha should yield lower |z| on new stable values."""
    from detector.config import DetectorConfig

    values = [2.0] * 5 + [40.0] * 10
    slow_last = feed(
        detector_cls(DetectorConfig(0.05, 3.0, 5, 2, 0.1)),
        "wh_stat_drift_slow",
        values,
    )[-1]
    fast_last = feed(
        detector_cls(DetectorConfig(0.75, 3.0, 5, 2, 0.1)),
        "wh_stat_drift_fast",
        values,
    )[-1]
    assert abs(fast_last.z_score) < abs(slow_last.z_score)


def test_spec_reference_on_statistical_matrix(detector_cls):
    """Full statistical equality on unpublished config matrix."""
    from detector.config import DetectorConfig
    from detector.types import MetricPoint

    cases = [
        (
            DetectorConfig(0.08, 3.2, 5, 2, 0.08),
            "wh_stat_m1",
            [1.0, 1.0, 1.0, 1.0, 1.0, 6.0, 6.0],
        ),
        (
            DetectorConfig(0.25, 2.0, 4, 3, 0.15),
            "wh_stat_m2",
            [20.0] * 4 + [20.0, 60.0, 60.0, 60.0],
        ),
    ]
    for cfg, stream_id, values in cases:
        agent = detector_cls(cfg)
        reference = ReferenceStreamingAnomalyDetector(cfg)
        for index, value in enumerate(values):
            point = MetricPoint(stream_id, f"2026-08-01T00:00:{index:02d}Z", value)
            got = agent.update(point)
            expected = reference.update(point)
            assert got.is_anomaly == expected.is_anomaly
            assert got.z_score == pytest.approx(expected.z_score, rel=1e-9, abs=1e-9)
            assert got.baseline_std == pytest.approx(expected.baseline_std, rel=1e-9, abs=1e-9)
            assert_result_invariants(got, threshold=cfg.z_threshold)


def test_rule_based_threshold_gaming_would_false_alarm_on_benign(detector_cls):
    """If agent uses naive rule value > X, benign noise would false-alarm."""
    import random

    cfg = default_config()
    detector = detector_cls(cfg)
    random.seed(4242)
    values = [100.0 + random.gauss(0, 0.1) for _ in range(30)]
    results = feed(detector, "wh_stat_ruletrap", values)
    naive_rule_fps = sum(1 for v, r in zip(values, results) if v > 100.5 and r.is_anomaly)
    assert sum(r.is_anomaly for r in results) == 0
    assert naive_rule_fps == 0 or math.isfinite(naive_rule_fps)
