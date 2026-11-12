#!/usr/bin/env bash
# CIR-258: the suites the change reads (`--changed` its base, the pre-mark tier, --quick) to $OUT/suites.json, then `repository`
# and `tools/verity/circuit_check` whole and fresh to $OUT/suites-whole.json (the latter is what the change adds to check's suite).
# BASE=1 runs only circuit_check's suite whole and fresh, with the three changed files as at the base.
set -uo pipefail
O=${OUT:-$RESEARCH_RUN_DIR/out}; mkdir -p "$O"
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
log() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$O/job.log"; }
git rev-parse HEAD > "$O/head.txt" 2>/dev/null; log "head $(cat "$O/head.txt")"
uv sync -q --all-packages --extra torch-cpu > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 2; }
if [ "${BASE:-0}" = 1 ]; then
  I=$RESEARCH_RUN_DIR/inputs
  cp "$I/base.targets.py" tools/verity/circuit_check/src/circuit_check/targets.py
  cp "$I/base.boolean_fp8.circuit-check.py" verity/definitions/registry/boolean_fp8.circuit-check.py
  cp "$I/base.boolean_fp8.circuit-check.json" verity/definitions/registry/boolean_fp8.circuit-check.json
  log "base: the three files as at 1885cf1fe08c; $(git diff --stat | tail -1)"
  s=$(date +%s)
  uv run tools/verity/check/suites.py tools/verity/circuit_check --fresh --mem-gb "$MEMGB" --json "$O/suites-whole.json" \
    --logs "$O/whole" > "$O/whole.log" 2>&1
  rc=$?; log "circuit_check whole at the base rc $rc in $(( $(date +%s) - s )) s"; tail -6 "$O/whole.log" | tee -a "$O/job.log"
  s=$(date +%s)
  VERITY_CIRCUIT_CHECK_SLOW=1 uv run tools/verity/check/suites.py tools/verity/circuit_check --fresh --mem-gb 56 \
    --json "$O/suites-walks.json" --logs "$O/walks" -- \
    -k "test_every_call_the_suite_checks_is_partitioned or test_a_call_whose_partition_is_unchecked_passes_call_status_to_its_callees" \
    > "$O/walks.log" 2>&1
  rc2=$?; log "walks at the base rc $rc2 in $(( $(date +%s) - s )) s"; tail -6 "$O/walks.log" | tee -a "$O/job.log"
  exit $(( rc || rc2 ))
fi
s=$(date +%s)
uv run tools/verity/check/suites.py --changed 1885cf1fe08c --quick --mem-gb "$MEMGB" --json "$O/suites.json" --logs "$O/suites" \
  > "$O/changed.log" 2>&1
rc=$?; log "--changed --quick rc $rc in $(( $(date +%s) - s )) s"; tail -8 "$O/changed.log" | tee -a "$O/job.log"
s=$(date +%s)
uv run tools/verity/check/suites.py repository tools/verity/circuit_check --fresh --mem-gb "$MEMGB" --json "$O/suites-whole.json" \
  --logs "$O/whole" > "$O/whole.log" 2>&1
rc2=$?; log "repository and circuit_check whole rc $rc2 in $(( $(date +%s) - s )) s"; tail -8 "$O/whole.log" | tee -a "$O/job.log"
exit $(( rc || rc2 ))
