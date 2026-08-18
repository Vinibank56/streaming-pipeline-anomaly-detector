"""Configuration for streaming anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectorConfig:
    """Tunable detector parameters."""

    ema_alpha: float
    z_threshold: float
    warmup_samples: int
    consecutive_required: int
    min_std: float


def default_config() -> DetectorConfig:
    """Production defaults used in visible verifier fixtures."""
    return DetectorConfig(
        ema_alpha=0.3,
        z_threshold=3.0,
        warmup_samples=5,
        consecutive_required=2,
        min_std=0.1,
    )
