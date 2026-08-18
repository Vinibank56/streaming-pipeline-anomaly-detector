"""Hidden edge-case tests for sparse, volatile, and multi-stream scenarios."""

from __future__ import annotations

import pytest

from helpers import default_config, feed, load_detector_module, make_point
from invariants import assert_result_invariants
from spec_reference import ReferenceStreamingAnomalyDetector


@pytest.fixture(scope="module")
def detector_cls():
    return load_detector_module().StreamingAnomalyDetector


def test_sparse_stream_never_alerts_before_warmup_complete(detector_cls):
    """Sparse data: fewer points than warmup_samples must never alert."""
    from detector.config import DetectorConfig

    cfg = DetectorConfig(0.3, 2.0, 10, 1, 0.1)
    results = feed(detector_cls(cfg), "wh_edge_sparse", [10.0, 999.0, 999.0, 999.0])
    assert all(not r.is_anomaly for r in results)
    assert all(not r.warmed_up for r in results)


def test_sparse_restart_after_reset_starts_cold(detector_cls):
    """After reset, stream must re-enter warmup and suppress alerts."""
    cfg = default_config()
    detector = detector_cls(cfg)
    feed(detector, "wh_edge_reset", [10.0] * 6 + [500.0, 500.0])
    detector.reset("wh_edge_reset")
    results = feed(detector, "wh_edge_reset", [10.0, 500.0, 500.0])
    assert results[0].warmed_up is False
    assert results[0].is_anomaly is False


def test_rapid_oscillation_does_not_false_alarm(detector_cls):
    """Volatile alternating spikes should not exceed FP budget with debouncing."""
    from detector.config import DetectorConfig

    cfg = DetectorConfig(0.2, 3.0, 5, 3, 0.1)
    pattern = [10.0, 50.0] * 8
    results = feed(detector_cls(cfg), "wh_edge_oscillate", [10.0] * 5 + pattern)
    assert sum(r.is_anomaly for r in results) == 0


def test_many_independent_streams_do_not_cross_contaminate(detector_cls):
    """100 one-point streams simulate concurrent ingestion without shared state."""
    cfg = default_config()
    detector = detector_cls(cfg)
    for index in range(100):
        result = detector.update(make_point(f"wh_edge_stream_{index:03d}", 0, float(index)))
        assert result.baseline_mean == float(index)
        assert result.is_anomaly is False
        assert result.warmed_up is False


def test_rapid_regime_change_matches_spec_reference(detector_cls):
    """Fast regime shifts verified against sealed reference on unpublished stream."""
    from detector.config import DetectorConfig
    from detector.types import MetricPoint

    cfg = DetectorConfig(0.4, 3.0, 4, 2, 0.1)
    values = [1.0, 1.0, 1.0, 1.0, 50.0, 50.0, 1.0, 50.0, 50.0]
    agent = detector_cls(cfg)
    reference = ReferenceStreamingAnomalyDetector(cfg)
    for index, value in enumerate(values):
        point = MetricPoint("wh_edge_rapid", f"2026-09-01T00:00:{index:02d}Z", value)
        got = agent.update(point)
        expected = reference.update(point)
        assert got.z_score == pytest.approx(expected.z_score, rel=1e-9, abs=1e-9)
        assert got.is_anomaly == expected.is_anomaly
        assert_result_invariants(got, threshold=cfg.z_threshold)


def test_first_point_on_stream_always_benign(detector_cls):
    """Edge: first observation initializes baseline, never alerts."""
    cfg = default_config()
    result = detector_cls(cfg).update(make_point("wh_edge_first", 0, 1e6))
    assert result.is_anomaly is False
    assert result.z_score == 0.0
    assert result.warmed_up is False
