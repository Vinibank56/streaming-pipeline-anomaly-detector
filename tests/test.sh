#!/usr/bin/env bash
set +e

mkdir -p /logs/verifier

python3 --version >/dev/null

python3 -m pytest \
  /tests/test_visible.py \
  /tests/test_hidden.py \
  /tests/test_behavior_hidden.py \
  /tests/test_statistical_hidden.py \
  -q
status=$?

if [ "$status" -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi

exit 0
