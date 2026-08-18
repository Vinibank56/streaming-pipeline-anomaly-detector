"""Shared helpers for anomaly detector verifier tests."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def app_root() -> Path:
    if Path("/app").exists():
        return Path("/app")
    return Path(__file__).resolve().parents[1] / "environment" / "app"


def load_detector_module():
    root = str(app_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    for module_name in (
        "detector.streaming",
        "detector.config",
        "detector.types",
        "detector.ema",
    ):
        if module_name in sys.modules:
            del sys.modules[module_name]
    return importlib.import_module("detector.streaming")


def default_config():
    from detector.config import default_config as _default_config

    return _default_config()


def make_point(stream_id: str, index: int, value: float) -> "MetricPoint":
    from detector.types import MetricPoint

    return MetricPoint(
        stream_id=stream_id,
        timestamp_iso=f"2026-03-01T00:00:{index:02d}Z",
        value=value,
    )


def feed(detector, stream_id: str, values: list[float]):
    from detector.types import MetricPoint

    results = []
    for index, value in enumerate(values):
        point = MetricPoint(
            stream_id=stream_id,
            timestamp_iso=f"2026-03-01T00:00:{index:02d}Z",
            value=value,
        )
        results.append(detector.update(point))
    return results
