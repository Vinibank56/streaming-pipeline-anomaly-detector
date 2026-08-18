"""Types for streaming metric ingestion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricPoint:
    """Single observation from a metrics stream."""

    stream_id: str
    timestamp_iso: str
    value: float


@dataclass(frozen=True)
class AnomalyResult:
    """Detection outcome for one metric point."""

    is_anomaly: bool
    z_score: float
    baseline_mean: float
    baseline_std: float
    consecutive_high: int
    warmed_up: bool
    audit: dict
