#!/usr/bin/env bash
# CMP-12 W7: the plain FP8 baselines beside PoUW η, on one leased RTX PRO 6000 (tier 2a; the fill queue keeps running).
# In one lease: the harness's autotuned FP8 E4M3 baselines (cuBLASLt 12.9 and 13, CUTLASS sm_120, the tree's own) at the
# headline shapes, from node 2's harness build of the current native sources; then cuBLASLt through torch._scaled_mm timed like
# eta.py at the batch curve; then ncu (--clock-control none) on one torch call a shape.
# Run from a clone of the pushed commit: research run ... --cwd clone --send job_base.sh -- bash "$RESEARCH_RUN_DIR/inputs/job_base.sh"
set -euo pipefail
export HB=/workspace/pouw/harness/build/05a44c565456c043 NCU=/usr/local/cuda-13.0/bin/ncu TPY=/workspace/jobs/venv312/bin/python
git rev-parse HEAD > "$OUT/eta_commit.txt"
gpu-lease 1 --wait --max-min 25 -- bash benchmarks/pouw_eta/lease_base.sh
ls -la "$OUT"
