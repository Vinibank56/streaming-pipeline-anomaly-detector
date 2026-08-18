"""Hidden behavioral and property tests."""

from __future__ import annotations

import pytest

from helpers import default_config, feed, load_detector_module
from invariants import assert_result_invariants
from spec_reference import ReferenceStreamingAnomalyDetector


@pytest.fixture(scope="module")
def detector_cls():
    return load_detector_module().StreamingAnomalyDetector


def test_consecutive_counter_resets_after_normal_point(detector_cls):
    from detector.config import DetectorConfig

    cfg = DetectorConfig(0.05, 3.0, 5, 2, 0.1)
    detector = detector_cls(cfg)
    results = feed(detector, "reset_consec", [10.0] * 5 + [100.0, 10.0, 100.0])
    assert results[-2].consecutive_high == 0
    assert results[-2].is_anomaly is False


def test_baseline_std_is_not_mean_magnitude(detector_cls):
    """Anti-exploit: z must not use abs(mean) as dispersion proxy."""
    cfg = default_config()
    detector = detector_cls(cfg)
    results = feed(detector, "z_check", [10.0] * 8)
    last = results[-1]
    assert last.baseline_mean == pytest.approx(10.0)
    assert last.baseline_std != abs(last.baseline_mean)
    assert last.baseline_std >= cfg.min_std


def test_high_alpha_adapts_faster_after_regime_shift(detector_cls):
    from detector.config import DetectorConfig

    slow = DetectorConfig(0.05, 3.0, 5, 2, 0.1)
    fast = DetectorConfig(0.7, 3.0, 5, 2, 0.1)
    values = [10.0] * 5 + [50.0] * 8
    slow_last = feed(detector_cls(slow), "slow", values)[-1]
    fast_last = feed(detector_cls(fast), "fast", values)[-1]
    assert abs(fast_last.z_score) < abs(slow_last.z_score)


def test_reference_matrix_unseen_configs(detector_cls):
    from detector.config import DetectorConfig
    from detector.types import MetricPoint

    matrix = [
        (DetectorConfig(0.12, 2.8, 4, 2, 0.05), "wh_matrix_a", [3.0, 3.0, 3.0, 3.0, 8.0, 8.0]),
        (DetectorConfig(0.2, 3.5, 6, 3, 0.2), "wh_matrix_b", [50.0] * 6 + [200.0, 200.0, 200.0]),
    ]
    for cfg, stream_id, values in matrix:
        agent = detector_cls(cfg)
        reference = ReferenceStreamingAnomalyDetector(cfg)
        for index, value in enumerate(values):
            point = MetricPoint(stream_id, f"2026-06-01T00:00:{index:02d}Z", value)
            got = agent.update(point)
            expected = reference.update(point)
            assert got.is_anomaly == expected.is_anomaly
            assert got.consecutive_high == expected.consecutive_high


def test_min_std_prevents_divide_by_zero(detector_cls):
    from detector.config import DetectorConfig

    cfg = DetectorConfig(0.3, 3.0, 2, 1, 0.5)
    detector = detector_cls(cfg)
    results = feed(detector, "flat", [7.0, 7.0, 20.0])
    assert results[-1].baseline_std >= cfg.min_std
    assert results[-1].z_score == pytest.approx((20.0 - 7.0) / cfg.min_std, rel=1e-6)


def test_warmed_up_implies_sample_count(detector_cls):
    cfg = default_config()
    detector = detector_cls(cfg)
    results = feed(detector, "count_check", [1.0] * 7)
    for r in results:
        assert_result_invariants(r, threshold=cfg.z_threshold)
        if r.audit["sample_count"] < cfg.warmup_samples:
            assert r.warmed_up is False
        else:
            assert r.warmed_up is True
