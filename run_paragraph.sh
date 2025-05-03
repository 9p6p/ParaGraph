#!/bin/bash

for rounds in 5; do
  echo "Running with $rounds rounds..."
  if MATCH_ROUNDS=$rounds SAVE_PATH_BASE="./indices/t2i-10M/t2i-test-" ./build/tests/test_t2iend > logs/run_t2i-r${rounds}.log 2>&1; then
    echo "Run completed successfully for $rounds rounds"
  else
    echo "ERROR: Run failed for $rounds rounds with exit code $?"
  fi
  sleep 5
done