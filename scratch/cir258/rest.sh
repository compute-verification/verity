#!/usr/bin/env bash
# CIR-258 carried head: the eight suites r20261113-165939-b78b's suites.py had not finished when its time ran out.
set -uo pipefail
O=${OUT:-$RESEARCH_RUN_DIR/out}; mkdir -p "$O"
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
log() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$O/job.log"; }
git rev-parse HEAD > "$O/head.txt" 2>/dev/null; log "head $(cat "$O/head.txt")"
uv sync -q --all-packages --extra torch-cpu > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 2; }
uv run tools/verity/check/suites.py benchmarks/numerical infra/ops tools/verity/circuit_check verity/circuits/evaluate verity/circuits \
  verity/commitments verity/commitments/windows verity/definitions --json "$O/suites.json" --logs "$O/suites" > "$O/suites.log" 2>&1
src=$?; log "suites rc $src"; grep -E "^(passed|failed)|^[0-9]+ suites" "$O/suites.log" | tee -a "$O/job.log"
exit $src
