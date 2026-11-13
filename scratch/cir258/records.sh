#!/usr/bin/env bash
# CIR-258: whether the circuit_check suite's extra time at the head is its catalog built in-session because no catalog record
# counts for an edited targets.py (conftest.py `_from_records`), not anything the new entries or spots check.
#   head:    records for this tree made first (`circuit-check --all --records-only`, into a cache of the run's own), then the suite
#            whole and fresh with them handed (CIRCUIT_CHECK_RECORDS), as the graph's test-file runner hands them.
set -uo pipefail
O=${OUT:-$RESEARCH_RUN_DIR/out}; mkdir -p "$O"
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
log() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$O/job.log"; }
I=$RESEARCH_RUN_DIR/inputs
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
git rev-parse HEAD > "$O/head.txt" 2>/dev/null; log "head $(cat "$O/head.txt")"
uv sync -q --all-packages --extra torch-cpu > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 2; }
F=(tools/verity/circuit_check/src/circuit_check/targets.py verity/definitions/registry/boolean_fp8.circuit-check.py
   verity/definitions/registry/boolean_fp8.circuit-check.json)
B=(base.targets.py base.boolean_fp8.circuit-check.py base.boolean_fp8.circuit-check.json)
suite() {  # name, then env assignments
  local name=$1; shift
  local s=$(date +%s)
  env "$@" uv run tools/verity/check/suites.py tools/verity/circuit_check --fresh --mem-gb "$MEMGB" --json "$O/suites-$name.json" \
    --logs "$O/$name" > "$O/$name.log" 2>&1
  local r=$?; log "$name: circuit_check whole rc $r in $(( $(date +%s) - s )) s"; grep -E "^(passed|failed)|UsageError" "$O/$name.log" | tee -a "$O/job.log"
  return $r
}
rc=0
git checkout -q -- "${F[@]}"; log "head: $(git diff --stat | tail -1)"
s=$(date +%s)
uv run python -m circuit_check --all --records-only -q --cache "$T/cc" --jobs "$JOBS" --mem-gb "$RECMEM" > "$O/records.log" 2>&1
r=$?; log "records for the head rc $r in $(( $(date +%s) - s )) s"; tail -3 "$O/records.log" | tee -a "$O/job.log"
suite head-records CIRCUIT_CHECK_RECORDS="$T/cc"; r=$?; rc=$(( rc || r ))
if [ $r != 0 ]; then
  (cd tools/verity/circuit_check && CIRCUIT_CHECK_RECORDS="$T/cc" uv run pytest -p no:xdist -x -q \
     "tests/test_metadata_files.py" > "$O/head-records-direct.log" 2>&1); log "direct pytest rc $?"; tail -15 "$O/head-records-direct.log" | tee -a "$O/job.log"
fi
exit $rc
