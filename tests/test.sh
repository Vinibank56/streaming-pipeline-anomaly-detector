#!/usr/bin/env bash
# Verifier entrypoint: grade agent/oracle output; always exit 0.
set +euo pipefail

mkdir -p /logs/verifier

export PYTHONPATH="/app${PYTHONPATH:+:$PYTHONPATH}"

python3 -m pytest \
  /tests/test_visible.py \
  /tests/test_hidden.py \
  /tests/test_behavior_hidden.py \
  /tests/test_statistical_hidden.py \
  /tests/test_edge_hidden.py \
  -q
status=$?

if [ "$status" -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi

exit 0
