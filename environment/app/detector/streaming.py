"""Streaming anomaly detector (broken seed implementation)."""

from __future__ import annotations

from detector.config import DetectorConfig
from detector.types import AnomalyResult, MetricPoint


class StreamingAnomalyDetector:
    """
    Online anomaly detector for pipeline metric streams.

    The seed implementation intentionally violates several ML-ops rules.
    """

    def __init__(self, config: DetectorConfig) -> None:
        self._config = config
        # BUG: single global baseline shared across all streams
        self._mean = 0.0
        self._count = 0
        self._consecutive = 0

    def update(self, point: MetricPoint) -> AnomalyResult:
        self._count += 1

        # BUG: simple cumulative mean instead of EMA (ignores concept drift / ema_alpha)
        prev_mean = self._mean
        self._mean = ((self._mean * (self._count - 1)) + point.value) / self._count

        # BUG: wrong z-score denominator uses mean magnitude, not std; no min_std floor
        if prev_mean == 0:
            z = 0.0
        else:
            z = (point.value - self._mean) / abs(self._mean)

        # BUG: no warmup gate — may alert immediately
        # BUG: single-point threshold breach, no consecutive_required debounce
        if abs(z) > self._config.z_threshold:
            self._consecutive += 1
            is_anomaly = True
        else:
            self._consecutive = 0
            is_anomaly = False

        return AnomalyResult(
            is_anomaly=is_anomaly,
            z_score=z,
            baseline_mean=self._mean,
            baseline_std=abs(self._mean),
            consecutive_high=self._consecutive,
            warmed_up=True,
            audit={
                "stream_id": point.stream_id,
                "sample_count": self._count,
                "ema_alpha_applied": self._config.ema_alpha,
                "threshold": self._config.z_threshold,
            },
        )

    def reset(self, stream_id: str | None = None) -> None:
        """Reset detector state. BUG: ignores stream_id because state is global."""
        self._mean = 0.0
        self._count = 0
        self._consecutive = 0
