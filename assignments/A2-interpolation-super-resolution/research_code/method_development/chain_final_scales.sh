#!/usr/bin/env bash
# Sequential final campaign (one memory-heavy job at a time). Re-runnable: finished runs are skipped.
set -euo pipefail
PY=/opt/anaconda3/envs/eda311/bin/python
MD="$(cd "$(dirname "$0")" && pwd)"
LOG="$MD/results/final_scales"
mkdir -p "$LOG"
cd "$MD/.."
st() { echo "[$(date '+%F %T')] $*" >> "$LOG/chain_status.log"; }

st "start x8 bsd"
$PY "$MD/run_final_scales.py" --scale 8 --pool bsd >> "$LOG/x8.log" 2>&1
POOL=$($PY "$MD/decide_pool.py" | tail -1)
st "pool decision: $POOL"
st "start x4 $POOL"
$PY "$MD/run_final_scales.py" --scale 4 --pool "$POOL" >> "$LOG/x4.log" 2>&1
st "start x2 $POOL"
$PY "$MD/run_final_scales.py" --scale 2 --pool "$POOL" >> "$LOG/x2.log" 2>&1
st "CHAIN DONE"
