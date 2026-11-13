#!/usr/bin/env bash
# CIR-258: tools/verity/circuit_check's suite whole and fresh, at the head, at the base (the three changed files as at 1885cf1fe08c),
# then at the head again, in one slot, so the before and after share a machine and its load.
set -uo pipefail
O=${OUT:-$RESEARCH_RUN_DIR/out}; mkdir -p "$O"
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
log() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$O/job.log"; }
I=$RESEARCH_RUN_DIR/inputs
git rev-parse HEAD > "$O/head.txt" 2>/dev/null; log "head $(cat "$O/head.txt")"
uv sync -q --all-packages --extra torch-cpu > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 2; }
F=(tools/verity/circuit_check/src/circuit_check/targets.py verity/definitions/registry/boolean_fp8.circuit-check.py
   verity/definitions/registry/boolean_fp8.circuit-check.json)
B=(base.targets.py base.boolean_fp8.circuit-check.py base.boolean_fp8.circuit-check.json)
rc=0
for step in head1 base head2; do
  if [ "$step" = base ]; then for i in 0 1 2; do cp "$I/${B[$i]}" "${F[$i]}"; done
  else git checkout -q -- "${F[@]}"; fi
  log "$step: $(git diff --stat | tail -1)"
  s=$(date +%s)
  uv run tools/verity/check/suites.py tools/verity/circuit_check --fresh --mem-gb "$MEMGB" --json "$O/suites-$step.json" \
    --logs "$O/$step" > "$O/$step.log" 2>&1
  r=$?; rc=$(( rc || r )); log "$step: circuit_check whole rc $r in $(( $(date +%s) - s )) s"; grep -E "^(passed|failed)" "$O/$step.log" | tee -a "$O/job.log"
done
exit $rc
