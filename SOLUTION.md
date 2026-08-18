# Reference Solution Verification

This document proves the oracle (`solution/`) is complete and achieves full reward.

## Implementation checklist

| Requirement | Implementation in `solution/streaming.py` |
|-------------|-------------------------------------------|
| Per-stream EMA baseline | `_StreamState` dict keyed by `stream_id` |
| Pre-update z-scoring | `ema.z_score()` called **before** `ema.ema_update()` |
| Concept drift via `ema_alpha` | `ema.ema_update(..., self._config.ema_alpha)` |
| Warmup gating | `warmed_up = sample_count >= warmup_samples`; no alert before |
| Consecutive debouncing | `consecutive_high` incremented/reset; `is_anomaly` requires `>= consecutive_required` |
| `min_std` floor | `ema.effective_std(state.var, min_std)` |
| Stream-scoped reset | `reset(stream_id)` pops one stream; `reset(None)` clears all |
| Complete audit block | All 6 audit fields on every result |

## Oracle verification command

```bash
cp solution/streaming.py environment/app/detector/streaming.py
PYTHONPATH=environment/app pytest tests/ -q
# Expected: all tests passed
```

## Broken seed (nop floor)

```bash
# Restore environment/app/detector/streaming.py to broken seed
PYTHONPATH=environment/app pytest tests/ -q
# Expected: majority fail
```

## Harness entrypoint

`solution/solve.sh` copies the reference file and runs all five pytest modules before grading.
