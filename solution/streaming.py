"""Streaming anomaly detector (reference implementation)."""

from __future__ import annotations

from dataclasses import dataclass

from detector import ema
from detector.config import DetectorConfig
from detector.types import AnomalyResult, MetricPoint


@dataclass
class _StreamState:
    mean: float = 0.0
    var: float = 0.0
    sample_count: int = 0
    consecutive_high: int = 0
    initialized: bool = False


class StreamingAnomalyDetector:
    """Online EMA baseline detector with warmup and consecutive-hit debouncing."""

    def __init__(self, config: DetectorConfig) -> None:
        self._config = config
        self._streams: dict[str, _StreamState] = {}

    def _state(self, stream_id: str) -> _StreamState:
        if stream_id not in self._streams:
            self._streams[stream_id] = _StreamState()
        return self._streams[stream_id]

    def update(self, point: MetricPoint) -> AnomalyResult:
        state = self._state(point.stream_id)

        if not state.initialized:
            state.mean = point.value
            state.var = 0.0
            state.sample_count = 1
            state.initialized = True
            return AnomalyResult(
                is_anomaly=False,
                z_score=0.0,
                baseline_mean=state.mean,
                baseline_std=ema.effective_std(state.var, self._config.min_std),
                consecutive_high=0,
                warmed_up=False,
                audit=self._audit(point.stream_id, state, False),
            )

        baseline_mean = state.mean
        baseline_std = ema.effective_std(state.var, self._config.min_std)
        z = ema.z_score(point.value, baseline_mean, baseline_std)

        if abs(z) > self._config.z_threshold:
            state.consecutive_high += 1
        else:
            state.consecutive_high = 0

        state.mean, state.var = ema.ema_update(
            state.mean,
            state.var,
            point.value,
            self._config.ema_alpha,
        )
        state.sample_count += 1

        warmed_up = state.sample_count >= self._config.warmup_samples
        is_anomaly = warmed_up and state.consecutive_high >= self._config.consecutive_required

        return AnomalyResult(
            is_anomaly=is_anomaly,
            z_score=z,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            consecutive_high=state.consecutive_high,
            warmed_up=warmed_up,
            audit=self._audit(point.stream_id, state, warmed_up),
        )

    def reset(self, stream_id: str | None = None) -> None:
        if stream_id is None:
            self._streams.clear()
        else:
            self._streams.pop(stream_id, None)

    def _audit(self, stream_id: str, state: _StreamState, warmed_up: bool) -> dict:
        return {
            "stream_id": stream_id,
            "sample_count": state.sample_count,
            "ema_alpha_applied": self._config.ema_alpha,
            "threshold": self._config.z_threshold,
            "warmed_up": warmed_up,
            "consecutive_required": self._config.consecutive_required,
        }
