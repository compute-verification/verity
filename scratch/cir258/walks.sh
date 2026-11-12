#!/usr/bin/env bash
# CIR-258: the two opt-in walks that hold targets.CALLEE_CALLS (test_circuit_check.py), at the head; then, as a negative run once
# and not kept, the second walk with the head's ScaledMmFp8BlockCoordinateGivenScale_v2 entry removed, which should fail naming
# the coordinate.  --cwd clone: the negative edits targets.py in the run's own clone, which is removed when the run ends.
set -uo pipefail
O=${OUT:-$RESEARCH_RUN_DIR/out}; mkdir -p "$O"
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
log() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$O/job.log"; }
git rev-parse HEAD > "$O/head.txt" 2>/dev/null; log "head $(cat "$O/head.txt")"
uv sync -q --all-packages --extra torch-cpu > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 2; }
WALKS="test_every_call_the_suite_checks_is_partitioned or test_a_call_whose_partition_is_unchecked_passes_call_status_to_its_callees"
log "walks at the head: start"
VERITY_CIRCUIT_CHECK_SLOW=1 uv run tools/verity/check/suites.py tools/verity/circuit_check --fresh --mem-gb 56 \
  --json "$O/suites-walks.json" --logs "$O/walks" -- -k "$WALKS" > "$O/walks.log" 2>&1
rc=$?
log "walks at the head: rc $rc"; tail -8 "$O/walks.log" | tee -a "$O/job.log"
T=tools/verity/circuit_check/src/circuit_check/targets.py
python3 - "$T" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
line = '    "ScaledMmFp8BlockCoordinateGivenScale_v2": ("ScaledMmFp8BlockSharedScale_v2",),\n'
assert s.count(line) == 1, "the entry is not in targets.py once"
open(p, "w").write(s.replace(line, ""))
PY
git diff --stat > "$O/negative.diff.stat"; git diff > "$O/negative.diff"
log "negative: entry removed ($(cat "$O/negative.diff.stat" | tail -1))"
VERITY_CIRCUIT_CHECK_SLOW=1 uv run tools/verity/check/suites.py tools/verity/circuit_check --fresh --mem-gb 56 \
  --json "$O/suites-negative.json" --logs "$O/negative" \
  -- -k test_a_call_whose_partition_is_unchecked_passes_call_status_to_its_callees > "$O/negative.log" 2>&1
nrc=$?
git checkout -- "$T"
log "negative: rc $nrc (expected nonzero)"; grep -h -o "set targets.CALLEE_CALLS to.\{0,200\}" -r "$O/negative" "$O/negative.log" | head -3 | tee -a "$O/job.log"
exit $rc
