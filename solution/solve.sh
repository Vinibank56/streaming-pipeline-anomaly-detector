#!/usr/bin/env bash
# Reference implementation for streaming-pipeline-anomaly-detector.
set -euo pipefail

cp /solution/streaming.py /app/detector/streaming.py

python3 -m pytest \
  /tests/test_visible.py \
  /tests/test_hidden.py \
  /tests/test_behavior_hidden.py \
  /tests/test_statistical_hidden.py \
  /tests/test_edge_hidden.py \
  -q
