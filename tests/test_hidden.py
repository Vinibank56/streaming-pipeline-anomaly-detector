"""Held-out edge cases not fully specified in instruction.md."""

from __future__ import annotations

import random

import pytest

from helpers import default_config, feed, load_detector_module
from invariants import assert_result_invariants
from spec_reference import ReferenceStreamingAnomalyDetector


@pytest.fixture(scope="module")
def detector_cls():
    return load_detector_module().StreamingAnomalyDetector


def test_per_stream_state_is_isolated(detector_cls):
    cfg = default_config()
    detector = detector_cls(cfg)
    detector.update(__import__("detector.types", fromlist=["MetricPoint"]).MetricPoint(
        "stream_a", "2026-04-01T00:00:00Z", 1000.0
    ))
    from helpers import make_point

    result = detector.update(make_point("stream_b", 0, 10.0))
    assert result.baseline_mean == 10.0
    assert result.warmed_up is False
    assert result.is_anomaly is False
    assert_result_invariants(result, threshold=cfg.z_threshold)


def test_benign_sequence_zero_false_positives(detector_cls):
    """Low false-positive requirement on noisy but stable metrics."""
    cfg = default_config()
    detector = detector_cls(cfg)
    random.seed(20260401)
    values = [10.0 + random.gauss(0, 0.05) for _ in range(40)]
    results = feed(detector, "benign_latency", values)
    assert sum(r.is_anomaly for r in results) == 0


def test_high_alpha_adapts_after_concept_drift(detector_cls):
    from detector.config import DetectorConfig

    cfg = DetectorConfig(
        ema_alpha=0.7,
        z_threshold=3.0,
        warmup_samples=4,
        consecutive_required=2,
        min_std=0.1,
    )
    detector = detector_cls(cfg)
    results = feed(detector, "drift_fast", [5.0] * 4 + [50.0] * 6)
    assert results[-1].is_anomaly is False
    assert results[-1].warmed_up is True


def test_reset_clears_stream_state(detector_cls):
    from helpers import make_point

    cfg = default_config()
    detector = detector_cls(cfg)
    feed(detector, "reset_me", [10.0] * 6)
    detector.reset("reset_me")
    result = detector.update(make_point("reset_me", 0, 42.0))
    assert result.baseline_mean == 42.0
    assert result.warmed_up is False
    assert result.consecutive_high == 0


@pytest.mark.parametrize(
    "stream_id,values,config_kwargs",
    [
        ("wh_unseen_x1", [2.0, 2.0, 2.0, 2.0, 2.0, 9.0, 9.0], {"ema_alpha": 0.05}),
        ("wh_unseen_x2", [100.0] * 6 + [100.0, 500.0], {"z_threshold": 2.5, "ema_alpha": 0.08}),
        ("wh_unseen_x3", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], {"warmup_samples": 3}),
    ],
)
def test_unpublished_streams_match_spec_reference(detector_cls, stream_id, values, config_kwargs):
    base = default_config()
    from detector.config import DetectorConfig

    cfg = DetectorConfig(
        ema_alpha=config_kwargs.get("ema_alpha", base.ema_alpha),
        z_threshold=config_kwargs.get("z_threshold", base.z_threshold),
        warmup_samples=config_kwargs.get("warmup_samples", base.warmup_samples),
        consecutive_required=config_kwargs.get("consecutive_required", base.consecutive_required),
        min_std=config_kwargs.get("min_std", base.min_std),
    )
    agent = detector_cls(cfg)
    reference = ReferenceStreamingAnomalyDetector(cfg)

    for index, value in enumerate(values):
        from detector.types import MetricPoint

        point = MetricPoint(stream_id, f"2026-05-01T00:00:{index:02d}Z", value)
        got = agent.update(point)
        expected = reference.update(point)
        assert got.is_anomaly == expected.is_anomaly
        assert got.z_score == pytest.approx(expected.z_score, rel=1e-9, abs=1e-9)
        assert got.baseline_mean == pytest.approx(expected.baseline_mean, rel=1e-9, abs=1e-9)
        assert got.consecutive_high == expected.consecutive_high
        assert got.warmed_up == expected.warmed_up


def test_uses_ema_not_cumulative_mean(detector_cls):
    """After drift, cumulative mean would stay biased; EMA should adapt."""
    from detector.config import DetectorConfig

    cfg = DetectorConfig(0.5, 3.0, 3, 2, 0.1)
    detector = detector_cls(cfg)
    results = feed(detector, "ema_check", [1.0, 1.0, 1.0, 100.0, 100.0, 100.0])
    assert results[-1].baseline_mean > 50.0
    assert results[-1].baseline_mean < 100.0
