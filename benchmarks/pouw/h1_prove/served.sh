#!/usr/bin/env bash
# The units a served H1 window's own draw coin selects (served.py), read from the window's run on the node that holds it:
#   research run --on vy-nebius-2 --project verity --campaign pouw --source . --cwd source --mem-gb 16 \
#     --declared-output 'out/*' --env H1_SERVED=/workspace/research/runs/RUN/out --env H1_PIN=SHA512 \
#     --env H1_DESCRIPTOR=/workspace/research/runs/PROGRAM_RUN/out/program/window-program.json.gz \
#     -- bash benchmarks/pouw/h1_prove/served.sh
# H1_SERVED: a vllm H1 window run's out/ (serve/pouw-h1-calls.json, gate/windows.jsonl, gate/fetch/<session>/out), or a
#   .tgz of those (--send it, then H1_SERVED=inputs/NAME.tgz); a relative path is the run dir's
# H1_PIN: the SHA-512 of the challenger's public key the verifier pins (never read from the run)
# H1_DESCRIPTOR: the window Program's descriptor (the rehearsal's window-program.json.gz), the circuit its period-open signed
# Then two negatives, each refused: another key's pin, and a draw root the coin didn't follow.
set -uo pipefail
export HZ_CONFIGS=served HZ_MEM_GB=16
# shellcheck source=../hidden_zk/job.sh
. benchmarks/pouw/hidden_zk/job.sh
D=${H1_SERVED:?} PIN=${H1_PIN:?} DESC=${H1_DESCRIPTOR:?}
case $D in /*) ;; *) D=$RESEARCH_RUN_DIR/$D ;; esac
[ -f "$D" ] && { mkdir -p "$W/served" && tar -xzf "$D" -C "$W/served" && D=$W/served; }
PY=$(cached venv numpy2.5.3-blake3-1.0.9 build_python)/bin/python
[ -x "$PY" ] || { log "build FAILED"; tail -20 "$OUT"/build-*.log; exit 1; }
export PYTHONPATH=$SRC:$SRC/integrations/vllm
S=$(ls -d "$D"/gate/fetch/*/out | head -1)
log "served draw: window run $D, session $S"
"$PY" "$SRC/benchmarks/pouw/h1_prove/served.py" --calls "$D/serve/pouw-h1-calls.json" --session "$S" \
  --ledger "$D/gate/windows.jsonl" --pin "$PIN" --descriptor "$DESC" --out "$OUT/drawn.json" 2>&1 | tee -a "$OUT/job.log"
rc=${PIPESTATUS[0]}
[ "$rc" = 0 ] || exit "$rc"
"$PY" - "$D/serve/pouw-h1-calls.json" "$W/forged-calls.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
r = d["commitment"]["draw"]["root"]
d["commitment"]["draw"]["root"] = ("0" if r[0] != "0" else "1") + r[1:]
json.dump(d, open(sys.argv[2], "w"))
EOF
neg() {
  local name=$1; shift
  if "$PY" "$SRC/benchmarks/pouw/h1_prove/served.py" --session "$S" --ledger "$D/gate/windows.jsonl" --descriptor "$DESC" "$@" \
      > "$OUT/neg-$name.log" 2>&1; then log "FAIL: $name drew units"; return 1; fi
  grep -q '^REFUSED' "$OUT/neg-$name.log" || { log "FAIL: $name did not end in a refusal"; tail -3 "$OUT/neg-$name.log"; return 1; }
  log "negative $name: $(grep '^REFUSED' "$OUT/neg-$name.log" | head -1)"
}
ok=0
neg other-pin --calls "$D/serve/pouw-h1-calls.json" --pin "$(printf '%0128d' 0)" || ok=1
neg other-draw-root --calls "$W/forged-calls.json" --pin "$PIN" || ok=1
exit $ok
