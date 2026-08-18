# Verification architecture (reviewer reference — not shown to agent)

This document explains how grading works. The agent only reads `instruction.md` and `/app`.

## Visible vs hidden (sealed from agent)

| Module | Agent visibility | Purpose |
|--------|------------------|---------|
| `test_visible.py` | **Visible** — scenarios documented in instruction.md | Self-check during development |
| `test_hidden.py` | **Hidden** — sealed under `/tests/` | Edge cases, FP rate, drift, spec-reference |
| `test_behavior_hidden.py` | **Hidden** | Property tests (consecutive reset, min_std) |
| `test_statistical_hidden.py` | **Hidden** | ML behavioral tests (detection sensitivity, EMA math) |
| `spec_reference.py` | **Hidden** | Canonical statistical implementation |
| `invariants.py` | **Hidden** | Objective constraints on every result |

The agent **cannot** read `/tests/` at runtime. Hidden modules use unpublished stream IDs (`wh_unseen_*`, `wh_matrix_*`, `wh_stat_*`) and compare outputs against `spec_reference.py` to block hard-coded responses.

## Multi-channel grading

1. **Functional literals** — expected z-scores and flags on held-out sequences
2. **Spec reference equality** — byte-for-byte statistical correctness vs sealed reference
3. **Invariants** — never alert before warmup; anomaly implies consecutive threshold met
4. **False-positive metric** — 40-point Gaussian noise sequence must yield 0 alerts
5. **Detection sensitivity** — sustained attack sequences must eventually alert
6. **Statistical identity** — z-score must match `ema.z_score` on pre-update baseline
7. **Concept drift** — different `ema_alpha` values must produce different baselines after regime shift

## Anti-gaming

Hard-coded `is_anomaly=True/False` fails because:
- Unpublished parametrized streams require exact z-score matches
- Benign sequence requires zero false positives
- Attack sequence requires true positive on sustained spikes
- Ignoring `ema_alpha` fails alpha-sensitivity and drift tests
