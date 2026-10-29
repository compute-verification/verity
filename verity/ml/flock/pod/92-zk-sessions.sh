#!/usr/bin/env bash
# The --zk sessions fixture of test_lean_zk.py, test_lean_hello.py and test_lean_session.py, on a CPU machine, on hidden_zk's
# builds (benchmarks/pouw/hidden_zk/job.sh, sourced: the selftest build of flock-circuit, with seed-injection, and the Lean
# flock-verify). From a clone:
#   research run --on auto --source . --as LANE --project verity --cwd clone --cpus 16 --mem-gb 32 \
#     -- bash verity/ml/flock/pod/92-zk-sessions.sh
# RoPE d64 (rope-head/d64/neox-bf16, generated n=32 seed 20260926) staged by verity_flock.circuit at 16 and 32 instances
# (one circuit, hidden outputs), then, all with the selftest build (statement verity/flock-circuit+seed-injection):
#   sessions/selftest   selftest --zk --record-dir on inst-16 (every case), converted by selftest_records.py
#   sessions/m26        serve/prove --zk on 32 instances, one table
#   sessions/j2         the same at --session-tables 2
#   sessions/drawn      serve --draw subset:16 on 32
# Kept in $OUT: fixture/ (stage/{circuit.txt,pub-16.bin,pub-32.bin} and sessions/, the tree to put as fixture/v1), lean.log
# (Lean's verdicts on the honest selftest session and each served one) and the logs; never an inst file.
set -uo pipefail
export HZ_CONFIGS=zk-sessions HZ_MEM_GB=${ZS_MEM_GB:-32}
[ -n "${ZS_CACHE:-}" ] && export HZ_CACHE=$ZS_CACHE
source benchmarks/pouw/hidden_zk/job.sh
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/benchmarks/numerical/python
STATEMENT=verity/flock-circuit+seed-injection
setup
FIX=$OUT/fixture
mkdir -p "$FIX/stage" "$FIX/sessions"

"$PY" -m verity_numerical.bench.generate --subcircuit rope-head/d64/neox-bf16 --n 32 --seed 20260926 --out "$W/sets" \
  > "$OUT/generate.log" 2>&1 || { log "generate FAILED: $(tail -2 "$OUT/generate.log")"; exit 1; }
SET=$(dirname "$(find "$W/sets" -name manifest.json | head -1)")
"$PY" -m verity_flock.circuit --input-set "$SET" --n 16 32 --out "$W/stage" > "$OUT/stage.json" 2> "$OUT/stage.err" \
  || { log "stage FAILED: $(tail -2 "$OUT/stage.err")"; exit 1; }
cp "$W/stage/circuit.txt" "$W/stage/pub-16.bin" "$W/stage/pub-32.bin" "$FIX/stage/"
log "staged: circuit_sha512 $(sha512sum "$W/stage/circuit.txt" | cut -c1-16)"
ST=(--circuit "$W/stage/circuit.txt" --tables "$W/stage/tables")

"$SBIN" selftest --zk --instances "$W/stage/inst-16.bin" "${ST[@]}" --forged "$W/stage/circuit-forged.txt" \
  --record-dir "$W/rec" > "$OUT/selftest.out" 2> "$OUT/selftest.err"
log "selftest exit $?: $(grep SELFTEST "$OUT/selftest.out" | cut -c1-300)"
"$PY" "$SRC/verity/ml/flock/verifier/selftest_records.py" "$W/rec" "$FIX/sessions/selftest" > "$OUT/selftest_records.log" 2>&1 \
  || { log "selftest_records FAILED: $(tail -2 "$OUT/selftest_records.log")"; exit 1; }
log "selftest cases: $(ls "$FIX/sessions/selftest" | wc -l)"

PIN=$(sha512sum "$W/stage/circuit.txt" | cut -d' ' -f1)
served() {   # served NAME SERVE_ARGS PROVE_ARGS: one loopback --zk session on 32 instances under $FIX/sessions/NAME
  local name=$1 sargs=$2 pargs=$3 port srv
  port=$(free_port)
  "$SBIN" serve --zk --listen 127.0.0.1:$port --out "$FIX/sessions/$name" --instances "$W/stage/pub-32.bin" "${ST[@]}" \
    --pin "$PIN" --operator loopback $sargs > "$OUT/serve-$name.log" 2>&1 &
  srv=$!
  until grep -q '^SERVING ' "$OUT/serve-$name.log" 2>/dev/null; do kill -0 $srv 2>/dev/null || break; sleep 1; done
  "$SBIN" prove --zk --verifier 127.0.0.1:$port --instances "$W/stage/inst-32.bin" "${ST[@]}" $pargs \
    > "$OUT/prove-$name.out" 2> "$OUT/prove-$name.err"
  log "$name: prove exit $?: $(grep -h '^LIVE' "$OUT/prove-$name.out" | cut -c1-200)"
  kill $srv 2>/dev/null; wait $srv 2>/dev/null
}
served m26 "" ""
served j2 "--session-tables 2" "--session-tables 2"
served drawn "--draw subset:16" ""

verify() {   # verify PUBLIC TABLES SESSION...
  local pub=$1 tables=$2; shift 2
  "$LEAN" verify --zk --statement "$STATEMENT" --circuit "$FIX/stage/circuit.txt" --public "$FIX/stage/$pub" \
    --session-tables "$tables" --session "$@" 2>&1 | grep '^VERDICT ' \
    | "$PY" -c "import json, sys; [print(v['accepted'], v['why']) for v in (json.loads(l[8:]) for l in sys.stdin)]"
}
one() { ls -d "$FIX/sessions/$1"/*/ | head -1; }
{
  echo "selftest/honest: $(verify pub-16.bin 1 "$FIX/sessions/selftest/honest")"
  echo "m26: $(verify pub-32.bin 1 "$(one m26)")"
  echo "j2: $(verify pub-32.bin 2 "$(one j2)")"
  echo "drawn: $(verify pub-32.bin 1 "$(one drawn)")"
  for s in m26 j2 drawn; do echo "$s index: $(cat "$FIX/sessions/$s/index.jsonl")"; done
} | tee "$OUT/lean.log" | tee -a "$OUT/job.log"
log "zk-sessions: done"
