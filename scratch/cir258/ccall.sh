#!/usr/bin/env bash
# CIR-258: circuit-check's step as check runs it (`--all -q --jobs B`, no cache), timed, for what the change adds to check.
# BASE=1 first puts the three files the change touches back as they are at the change's base (the tree is then the base's).
set -uo pipefail
O=${OUT:-$RESEARCH_RUN_DIR/out}; mkdir -p "$O"
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
log() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$O/job.log"; }
git rev-parse HEAD > "$O/head.txt" 2>/dev/null; log "head $(cat "$O/head.txt")"
if [ "${BASE:-0}" = 1 ]; then
  I=$RESEARCH_RUN_DIR/inputs
  cp "$I/base.targets.py" tools/verity/circuit_check/src/circuit_check/targets.py
  cp "$I/base.boolean_fp8.circuit-check.py" verity/definitions/registry/boolean_fp8.circuit-check.py
  cp "$I/base.boolean_fp8.circuit-check.json" verity/definitions/registry/boolean_fp8.circuit-check.json
  log "base: the three files as at 1885cf1fe08c; $(git diff --stat | tail -1)"
fi
uv sync -q --all-packages --extra torch-cpu > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 2; }
log "synced; circuit-check --all start (jobs ${JOBS}, mem-gb ${MEMGB})"
s=$(date +%s)
uv run python -m circuit_check --all -q --jobs "$JOBS" --mem-gb "$MEMGB" --out "$O/circuit-check.json" > "$O/circuit-check.log" 2>&1
rc=$?
log "circuit-check --all rc $rc in $(( $(date +%s) - s )) s"
tail -12 "$O/circuit-check.log" | tee -a "$O/job.log"
exit 0
