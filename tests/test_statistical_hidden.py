"""Hidden statistical and ML-behavior tests — sealed from the agent."""

from __future__ import annotations

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
    agent = detector_cls(cfg)
    reference = ReferenceStreamingAnomalyDetector(cfg)
    history = [5.0, 5.0, 5.0, 5.0, 7.5]
    for index, value in enumerate(history):
        point = MetricPoint("wh_stat_z", f"2026-07-01T00:00:{index:02d}Z", value)
        agent.update(point)
        reference.update(point)

    probe = MetricPoint("wh_stat_z", "2026-07-01T00:00:10Z", 9.0)
    state_before = reference._state("wh_stat_z")  # noqa: SLF001
    expected_z = ema.z_score(
        probe.value,
        state_before.mean,
        ema.effective_std(state_before.var, cfg.min_std),
    )
    result = agent.update(probe)
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
    assert sum(r.is_anomaly for r in results) == 0


@pytest.mark.parametrize("seed", [101, 202, 303, 404])
def test_false_positive_rate_zero_across_seeds(detector_cls, seed):
    """FP rate must remain 0 under multiple benign noise seeds."""
    import random

    cfg = default_config()
    detector = detector_cls(cfg)
    random.seed(seed)
    values = [25.0 + random.gauss(0, 0.12) for _ in range(40)]
    results = feed(detector, f"wh_stat_fp_{seed}", values)
    assert sum(r.is_anomaly for r in results) == 0


def test_baseline_variance_increases_with_spread(detector_cls):
    """Statistical property: wider values produce larger baseline_std after warmup."""
    from detector.config import DetectorConfig

    cfg = DetectorConfig(0.3, 5.0, 3, 2, 0.1)
    tight = feed(detector_cls(cfg), "wh_stat_tight", [10.0, 10.1, 9.9, 10.0, 10.2])[-1]
    wide = feed(detector_cls(cfg), "wh_stat_wide", [10.0, 15.0, 5.0, 14.0, 6.0])[-1]
    assert wide.baseline_std > tight.baseline_std


def test_full_stream_statistical_equality_vs_reference(detector_cls):
    """Every step must match sealed reference — not just final output."""
    from detector.config import DetectorConfig
    from detector.types import MetricPoint

    cfg = DetectorConfig(0.12, 2.8, 5, 2, 0.05)
    values = [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 25.0, 25.0, 8.0]
    agent = detector_cls(cfg)
    reference = ReferenceStreamingAnomalyDetector(cfg)
    for index, value in enumerate(values):
        point = MetricPoint("wh_stat_full", f"2026-10-01T00:00:{index:02d}Z", value)
        got = agent.update(point)
        expected = reference.update(point)
        assert got.is_anomaly == expected.is_anomaly
        assert got.z_score == pytest.approx(expected.z_score, rel=1e-9, abs=1e-9)
        assert got.baseline_mean == pytest.approx(expected.baseline_mean, rel=1e-9, abs=1e-9)
        assert got.baseline_std == pytest.approx(expected.baseline_std, rel=1e-9, abs=1e-9)
        assert got.consecutive_high == expected.consecutive_high
        assert got.warmed_up == expected.warmed_up


def test_z_score_near_zero_when_value_equals_baseline(detector_cls):
    """When value matches baseline mean, |z| must be ~0."""
    cfg = default_config()
    results = feed(detector_cls(cfg), "wh_stat_equal", [12.0] * 10)
    assert results[-1].z_score == pytest.approx(0.0, abs=1e-6)


def test_concept_drift_fast_alpha_tracks_new_regime_closer(detector_cls):
    """After shift to 30.0, fast alpha baseline must be closer to 30 than slow."""
    from detector.config import DetectorConfig

    values = [10.0] * 6 + [30.0] * 8
    slow = feed(
        detector_cls(DetectorConfig(0.05, 3.0, 5, 2, 0.1)),
        "wh_stat_track_slow",
        values,
    )[-1]
    fast = feed(
        detector_cls(DetectorConfig(0.9, 3.0, 5, 2, 0.1)),
        "wh_stat_track_fast",
        values,
    )[-1]
    assert abs(fast.baseline_mean - 30.0) < abs(slow.baseline_mean - 30.0)
