# Repair the streaming pipeline anomaly detector

You are on the observability team for a real-time data pipeline at `/app`. The platform ingests millions of metric points per minute and must flag anomalies without flooding on-call with false positives. SRE found that the online detector fires too early, crosses streams, and ignores concept drift settings.

Fix **only** `/app/detector/streaming.py`. Modules `config.py`, `types.py`, and `ema.py` are correct — do not modify them.

## Background

```
Metric producers ──► StreamingAnomalyDetector.update() ──► Alert manager
                           ▲
                    DetectorConfig (EMA α, z threshold,
                    warmup, consecutive hits, min_std)
```

The detector maintains an **EMA baseline** per `stream_id`. Concept drift is handled by `ema_alpha`: higher α adapts faster to regime shifts; lower α retains history. Operations require:

1. **Warmup** — no alerts until enough samples build a baseline.
2. **Low false positives** — require consecutive threshold breaches, not single spikes.
3. **Correct scoring** — z-scores use standard deviation with a numeric floor, not mean magnitude.
4. **Stream isolation** — each `stream_id` has independent state.

## Statistical model (ML requirements)

Implement **online statistical anomaly detection** using:

| Component | Specification |
|-----------|---------------|
| **Baseline model** | Exponential moving average (EMA) of mean and variance via `ema.ema_update()` |
| **Anomaly score** | z-score = `(value - baseline_mean) / effective_std` computed **before** updating the baseline |
| **Concept drift** | Controlled by `DetectorConfig.ema_alpha` — higher α adapts faster to regime shifts |
| **False-positive control** | Require `consecutive_required` threshold breaches; suppress alerts during `warmup_samples` |
| **Dispersion floor** | Use `ema.effective_std(var, min_std)` — never divide by zero or use mean magnitude as std |

This is a statistical process-control detector (not deep learning), representative of production ML-ops pipelines.

### Evaluation metrics (graded on hidden suite)

- **False-positive rate**: 0 alerts on 50-point Gaussian noise around a stable mean (default config)
- **Detection sensitivity**: Must alert on sustained post-warmup spike sequences (unpublished configs)
- **Drift responsiveness**: Different `ema_alpha` values must produce measurably different baselines after regime shift
- **Statistical correctness**: z-scores must match `ema.z_score()` on the pre-update baseline

## Requirements

Implement `StreamingAnomalyDetector` in `/app/detector/streaming.py`:

```python
class StreamingAnomalyDetector:
    def __init__(self, config: DetectorConfig) -> None: ...
    def update(self, point: MetricPoint) -> AnomalyResult: ...
    def reset(self, stream_id: str | None = None) -> None: ...
```

### Update algorithm

1. **First point** for a stream: initialize baseline mean to `value`, variance to `0`, `sample_count=1`, return `is_anomaly=False`, `z_score=0`, `warmed_up=False`.

2. **Subsequent points** (compare before updating baseline):
   - `baseline_std = effective_std(ema_var, min_std)` from `ema.py`
   - `z_score = z_score(value, baseline_mean, baseline_std)` from `ema.py`
   - If `abs(z_score) > z_threshold`: increment `consecutive_high`, else reset to `0`
   - Update baseline with `ema_update(prev_mean, prev_var, value, ema_alpha)`
   - Increment `sample_count`
   - `warmed_up = sample_count >= warmup_samples`
   - `is_anomaly = warmed_up and consecutive_high >= consecutive_required`

3. **Reset**: clear one stream if `stream_id` provided, else clear all streams.

### Audit block (every result)

```python
{
    "stream_id": str,
    "sample_count": int,
    "ema_alpha_applied": float,
    "threshold": float,
    "warmed_up": bool,
    "consecutive_required": int,
}
```

### Behavioral invariants

- Never alert before warmup completes.
- Never alert on a single threshold breach when `consecutive_required > 1`.
- Baseline std must use `effective_std`, not `abs(mean)`.
- Streams must not leak state into each other.

## Constraints

- Edit only `/app/detector/streaming.py`.
- Use helpers in `ema.py`; do not reimplement EMA incorrectly.
- Do not hard-code stream IDs or outputs.

## Success metrics

1. **Statistical** — EMA baseline, pre-update z-scoring, and drift via `ema_alpha` implemented correctly.
2. **Visible suite** — `pytest /tests/test_visible.py` passes (self-check scenarios below).
3. **Hidden suite** — sealed tests verify FP rate, detection sensitivity, drift behavior, and full equality against the statistical contract on unpublished streams (you cannot inspect these tests).

> **Note:** Only `test_visible.py` scenarios are documented here. Hidden modules (`test_hidden.py`, `test_behavior_hidden.py`, `test_statistical_hidden.py`) and `spec_reference.py` are sealed from the agent.

## Self-check examples (visible tests)

**Slow-adapt config for spike tests:** `ema_alpha=0.05`, `z_threshold=3.0`, `warmup_samples=5`, `consecutive_required=2`, `min_std=0.1`.

| Scenario | Input sequence | Expected |
|----------|------------------|----------|
| Warmup | stream `warm_cpu`, ten `[10.0]` × 5 | all `is_anomaly=False`; point 5 has `warmed_up=True` |
| Single spike | `[10.0]×5 + [100.0]` | last: `is_anomaly=False`, `consecutive_high=1`, `z_score > 3` |
| Double spike | `[10.0]×5 + [100.0, 100.0]` | last: `is_anomaly=True`, `consecutive_high=2` |
| Stable | default config, `[10.0]×8` | last: `z_score≈0`, `is_anomaly=False` |

## Done when

`StreamingAnomalyDetector` satisfies the contract and passes visible, hidden, and behavioral verifier suites.
