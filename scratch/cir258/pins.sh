#!/usr/bin/env bash
# CIR-258: circuit-check's pins update pass on the Boolean FP8 sidecar's RANGES spots (the block-scaled FP8 GEMM at the four
# served widths and its coordinate at their K) and the coordinate's two pinned bindings, now held as a Call; no --as-call.
# The pins diff it makes is written to $OUT/pins.diff, since a clone is removed when the run ends.
set -uo pipefail
O=${OUT:-$RESEARCH_RUN_DIR/out}; mkdir -p "$O"
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
log() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$O/job.log"; }
git rev-parse HEAD > "$O/head.txt" 2>/dev/null; log "head $(cat "$O/head.txt")"
uv sync -q --all-packages --extra torch-cpu > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 2; }
log "synced"
uv run circuit-check 'ScaledMmFp8BlockCoordinateGivenScale_v2{K=128,G=128}' 'ScaledMmFp8BlockCoordinateGivenScale_v2{K=64,G=32}' \
  --ranges ScaledMmFp8BlockSharedScale_v2 ScaledMmFp8BlockCoordinateGivenScale_v2 --update-pins --mem-gb 40 \
  --out "$O/update.json" > "$O/update.log" 2>&1
rc=$?
git status --porcelain > "$O/status.txt"; git diff > "$O/pins.diff"
log "update rc $rc; files changed: $(wc -l < "$O/status.txt")"
tail -15 "$O/update.log" | tee -a "$O/job.log"
exit $rc
