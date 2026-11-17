#!/usr/bin/env bash
# A deferred window's audit on the node `run.sh copy` sent it to (`run.sh audit`), its compose hand-off then packed.
set -uo pipefail
IN=${RESEARCH_RUN_DIR:?run under research run}/inputs OUT=$RESEARCH_RUN_DIR/out
mkdir -p "$OUT"
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$OUT/audit-job.log"; }
D=/dev/shm/vllm-served-run/${SERVED_RUN:?start it with \`run.sh audit\`}
[ -f "$D/private/deferred/deferred.env" ] || { log "$D holds no deferral (\`run.sh copy\` sends it): nothing run"; exit 2; }
if [ -f "$IN/main.txt" ]; then
  MAIN_FILE=$IN/main.txt PP_CHALLENGER_KEY=$PWD/$PP_CHALLENGER_KEY PP_GATE_KEY=${PP_GATE_KEY/#\~/$HOME}
  export MAIN_FILE PP_CHALLENGER_KEY PP_GATE_KEY
fi
export D OUT
bash benchmarks/one_stage/served_verdicts.sh v3c > "$OUT/served_verdicts.log" 2>&1
rc=$?
log "served_verdicts.sh v3c exit $rc; its lines in out/verdicts.txt"
. adapters/vllm/tools/served_run/handoff.sh
pack_handoff
h=$?
[ "$h" != 4 ] || exit 4
[ "$rc" = 2 ] && [ "$h" = 0 ] && { log "pending: composed off this node: \`run.sh compose\` with this run's out/handoff/"; exit 0; }
[ "$rc" != 0 ] || rc=4
exit "$rc"
