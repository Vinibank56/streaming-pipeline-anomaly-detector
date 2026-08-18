"""Visible verifier cases mirrored in instruction.md for agent self-check."""

from __future__ import annotations

import pytest

from helpers import default_config, feed, load_detector_module
from invariants import assert_result_invariants


@pytest.fixture(scope="module")
def detector_cls():
    return load_detector_module().StreamingAnomalyDetector


def test_result_shape(detector_cls):
    detector = detector_cls(default_config())
    results = feed(detector, "shape_stream", [10.0, 10.0])
    result = results[-1]
    assert hasattr(result, "is_anomaly")
    assert hasattr(result, "z_score")
    assert hasattr(result, "audit")
    assert set(result.audit.keys()) == {
        "stream_id",
        "sample_count",
        "ema_alpha_applied",
        "threshold",
        "warmed_up",
        "consecutive_required",
    }


def test_warmup_suppresses_alerts(detector_cls):
    detector = detector_cls(default_config())
    results = feed(detector, "warm_cpu", [10.0] * 5)
    assert all(not r.is_anomaly for r in results)
    assert results[0].warmed_up is False
    assert results[3].warmed_up is False
    assert results[4].warmed_up is True
    for r in results:
        assert_result_invariants(r, threshold=default_config().z_threshold)


def test_single_spike_not_anomaly_until_consecutive(detector_cls):
    from detector.config import DetectorConfig

    cfg = DetectorConfig(
        ema_alpha=0.05,
        z_threshold=3.0,
        warmup_samples=5,
        consecutive_required=2,
        min_std=0.1,
    )
    detector = detector_cls(cfg)
    results = feed(detector, "spike_once", [10.0] * 5 + [100.0])
    last = results[-1]
    assert last.is_anomaly is False
    assert last.consecutive_high == 1
    assert last.z_score > cfg.z_threshold


def test_consecutive_spikes_trigger_anomaly(detector_cls):
    from detector.config import DetectorConfig

    cfg = DetectorConfig(
        ema_alpha=0.05,
        z_threshold=3.0,
        warmup_samples=5,
        consecutive_required=2,
        min_std=0.1,
    )
    detector = detector_cls(cfg)
    results = feed(detector, "spike_twice", [10.0] * 5 + [100.0, 100.0])
    last = results[-1]
    assert last.is_anomaly is True
    assert last.consecutive_high == 2
    assert last.z_score > cfg.z_threshold


def test_stable_baseline_near_zero_z_score(detector_cls):
    detector = detector_cls(default_config())
    results = feed(detector, "stable", [10.0] * 8)
    assert results[-1].z_score == pytest.approx(0.0, abs=1e-6)
    assert results[-1].is_anomaly is False
