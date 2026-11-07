#!/usr/bin/env bash
# Whole experiment on a Kaggle T4: data, synthesis, training, evaluation.
# Needs SARVAM_API_KEY and HF_TOKEN in the environment, a GPU and internet.
# Every step resumes, so rerunning after a failure picks up where it stopped.
set -euo pipefail
: "${SARVAM_API_KEY:?set SARVAM_API_KEY}"
MINUTES=${MINUTES:-10}
START=$SECONDS
lap() { echo "[$((SECONDS - START)) s] $1"; }

pip install -q -e .
lap "installed"

synthdet prepare --n-train 800 --n-test 200
lap "FLEURS sampled"

# Bulbul is network bound and MMS is GPU bound, so run them together.
synthdet synth bulbul --splits test --concurrency 2 &
BULBUL=$!
synthdet synth mms --splits train test
wait "$BULBUL"
lap "synthesis done"

synthdet train --minutes "$MINUTES"
lap "training done"

synthdet eval
lap "all done"
mkdir -p results
echo "{\"total_seconds\": $((SECONDS - START))}" > results/timing.json
