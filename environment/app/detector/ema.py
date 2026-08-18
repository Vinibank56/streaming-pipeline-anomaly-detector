"""EMA helpers for online baseline tracking with concept-drift responsiveness."""

from __future__ import annotations

import math


def ema_update(prev_mean: float, prev_var: float, value: float, alpha: float) -> tuple[float, float]:
    """
    Update exponential moving mean and variance.

    Uses the prior mean when computing the squared deviation so incoming
    points are compared against the pre-update baseline.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")

    new_mean = alpha * value + (1.0 - alpha) * prev_mean
    new_var = alpha * (value - prev_mean) ** 2 + (1.0 - alpha) * prev_var
    return new_mean, max(new_var, 0.0)


def effective_std(ema_var: float, min_std: float) -> float:
    """Convert EMA variance to a standard deviation with a numeric floor."""
    return max(math.sqrt(ema_var), min_std)


def z_score(value: float, baseline_mean: float, baseline_std: float) -> float:
    """Standard score of value against the current baseline."""
    if baseline_std <= 0:
        return 0.0
    return (value - baseline_mean) / baseline_std
