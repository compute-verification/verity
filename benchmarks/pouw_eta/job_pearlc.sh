#!/usr/bin/env bash
# CMP-12 W7: PoUW η of Pearl-C sm120 v1 (-h3) on one leased RTX PRO 6000 (tier 2a diagnostic; the fill queue keeps running).
# A: build.sh's ship (nvcc 13.0, SASS-gated, fixture) off the lease.  B, in one lease: run.py's fixture check on the card (the
# gate: nothing below runs unless it passes), run.py's split, eta.py's schedules and batch curve, the clock hold, then ncu
# (--clock-control none: the node's lock stays as it is) kernel by kernel and graph by graph.  CSVs from the reports after.
# ETA_LEASE names B's script instead (lease_pearlc2.sh: the SHA-512 probe, par<c>, the power holds).
# Run from a clone of the pushed commit: research run ... --cwd clone --send job_pearlc.sh -- bash "$RESEARCH_RUN_DIR/inputs/job_pearlc.sh"
set -euo pipefail
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
nice -n 19 bash benchmarks/pouw/pearl_c_sm120/build.sh "$d/ship"
cp "$d/ship/toolchain.txt" "$d/ship/source.txt" "$d/ship/sass.json" "$d/ship/MANIFEST.sha256" "$OUT/"
cp benchmarks/pouw_eta/eta.py "$d/ship/"
git rev-parse HEAD > "$OUT/eta_commit.txt"
export SHIP="$d/ship" NCU=/usr/local/cuda-13.0/bin/ncu
gpu-lease 1 --wait --max-min 25 -- bash "benchmarks/pouw_eta/${ETA_LEASE:-lease_pearlc.sh}"
for r in ncu_kernels ncu_graphs; do
  [ -f "$OUT/$r.ncu-rep" ] && "$NCU" --import "$OUT/$r.ncu-rep" --csv --page raw > "$OUT/$r.csv" || echo "no $r report" >&2
done
ls -la "$OUT"
