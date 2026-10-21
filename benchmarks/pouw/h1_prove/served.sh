#!/usr/bin/env bash
# The units a served H1 window's own draw coin selects (served.py), read from the window's run on the node that holds it:
#   research run --on vy-nebius-2 --project verity --campaign pouw --source . --cwd source --mem-gb 16 \
#     --declared-output 'out/*' --env H1_SERVED=/workspace/research/runs/RUN/out --env H1_PIN=SHA512 \
#     -- bash benchmarks/pouw/h1_prove/served.sh
# H1_SERVED: a vllm H1 window run's out/ (serve/pouw-h1-calls.json, gate/windows.jsonl, gate/fetch/<session>/out), or a
#   .tgz of those (--send it, then H1_SERVED=inputs/NAME.tgz); a relative path is the run dir's
# H1_PIN: the SHA-512 of the challenger's public key the verifier pins (never read from the run)
# H1_DRAW (1): the units to draw
set -uo pipefail
export HZ_CONFIGS=served HZ_MEM_GB=16
# shellcheck source=../hidden_zk/job.sh
. benchmarks/pouw/hidden_zk/job.sh
D=${H1_SERVED:?} PIN=${H1_PIN:?}
case $D in /*) ;; *) D=$RESEARCH_RUN_DIR/$D ;; esac
[ -f "$D" ] && { mkdir -p "$W/served" && tar -xzf "$D" -C "$W/served" && D=$W/served; }
LEAN=$(cached lean "$(tree_key verity/core/service/flock)" build_lean)/.lake/build/bin/flock-verify
PY=$(cached venv numpy2.5.3-blake3-1.0.9 build_python)/bin/python
[ -x "$LEAN" ] && [ -x "$PY" ] || { log "build FAILED"; tail -20 "$OUT"/build-*.log; exit 1; }
export PYTHONPATH=$SRC:$SRC/integrations/vllm
S=$(ls -d "$D"/gate/fetch/*/out | head -1)
log "served draw: window run $D, session $S"
"$PY" "$SRC/benchmarks/pouw/h1_prove/served.py" --calls "$D/serve/pouw-h1-calls.json" --session "$S" \
  --ledger "$D/gate/windows.jsonl" --pin "$PIN" --lean "$LEAN" --k "${H1_DRAW:-1}" --out "$OUT/drawn.json" 2>&1 | tee -a "$OUT/job.log"
exit "${PIPESTATUS[0]}"
