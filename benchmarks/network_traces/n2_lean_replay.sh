#!/usr/bin/env bash
# n2_lean_replay.py over recorded n2_zk.sh runs (or chain_record.sh's warden-path runs, their N2 under out/n2), read in
# place on the machine whose runner holds their run dirs, with N2's network-records built in the shipped tree
# (`--cwd clone`):
#   research run --on vy-nebius-cpu-1 --pool research --project verity --cwd clone -- \
#     bash benchmarks/network_traces/n2_lean_replay.sh RUN_ID ...
# env: N2_RUNS_DIR (where the run dirs are; default the runner's, beside this run's)  JOBS (runs replayed at once, 4)
# Kept in $OUT: replay.json, replay.log, the build's log and the tree's head.
set -uo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:?run this under research run}/out}
RUNS=${N2_RUNS_DIR:-${RESEARCH_RUN_DIR%/*}}
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/head.txt" 2>/dev/null || true
export PATH=$HOME/.elan/bin:$PATH
(cd verity/core/protocols/network/lean && lake build network-records) > "$OUT/build-network-records.log" 2>&1 \
  || { tail -40 "$OUT/build-network-records.log"; exit 7; }
export VERITY_NETWORK_RECORDS_EXE=$PWD/verity/core/protocols/network/lean/.lake/build/bin/network-records
dirs=()
for r in "$@"; do dirs+=("$RUNS/$r"); done
uv run -q --locked python benchmarks/network_traces/n2_lean_replay.py --out "$OUT" --jobs "${JOBS:-4}" "${dirs[@]}" \
  2>&1 | tee "$OUT/replay.log"
exit "${PIPESTATUS[0]}"
