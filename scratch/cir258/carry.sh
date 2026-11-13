#!/usr/bin/env bash
# CIR-258 carried head: pins.py --regen at the root (its diff to $OUT/pins-regen.diff), then circuit-check on the Boolean FP8
# sidecar's RANGES spots and the coordinate's two catalog bindings (no --as-call, no pin update), then suites.py (the suites
# whose inputs changed). The last two run only if --regen rewrote nothing.
set -uo pipefail
O=${OUT:-$RESEARCH_RUN_DIR/out}; mkdir -p "$O"
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
log() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$O/job.log"; }
git rev-parse HEAD > "$O/head.txt" 2>/dev/null; log "head $(cat "$O/head.txt")"
python3 tools/verity/check/pins.py --stale > "$O/pins-stale.log" 2>&1; log "pins --stale rc $?"; tail -5 "$O/pins-stale.log" | tee -a "$O/job.log"
python3 tools/verity/check/pins.py --regen > "$O/pins-regen.log" 2>&1; prc=$?
git status --porcelain > "$O/pins-status.txt"; git diff --binary > "$O/pins-regen.diff"
log "pins --regen rc $prc; files changed: $(wc -l < "$O/pins-status.txt")"; tail -20 "$O/pins-regen.log" | tee -a "$O/job.log"
if [ -s "$O/pins-status.txt" ] || [ $prc != 0 ]; then log "stopping: pins rewrote something or stayed stale"; exit 3; fi
uv sync -q --all-packages --extra torch-cpu > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 2; }
log "synced"
uv run circuit-check 'ScaledMmFp8BlockCoordinateGivenScale_v2{K=128,G=128}' 'ScaledMmFp8BlockCoordinateGivenScale_v2{K=64,G=32}' \
  --ranges ScaledMmFp8BlockSharedScale_v2 ScaledMmFp8BlockCoordinateGivenScale_v2 --no-cache --mem-gb 40 \
  --out "$O/circuit-check.json" > "$O/circuit-check.log" 2>&1
crc=$?; log "circuit-check rc $crc; files changed: $(git status --porcelain | wc -l)"; tail -15 "$O/circuit-check.log" | tee -a "$O/job.log"
uv run tools/verity/check/suites.py --json "$O/suites.json" --logs "$O/suites" > "$O/suites.log" 2>&1
src=$?; log "suites rc $src"; tail -12 "$O/suites.log" | tee -a "$O/job.log"
exit $(( crc || src ))
