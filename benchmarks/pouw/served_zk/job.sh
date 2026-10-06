#!/usr/bin/env bash
# served_zk's verifier side on node 2's CPUs: a hidden tile proof of one served call, against that call's commitment in the
# ZK statement's format, after serve.sh's run SERVED_RUN retained its two requests under $WORK/passes/$SERVED_RUN.
#   research run --on vy-nebius-2 --project verity --campaign pouw --source <tree> --cwd source --declared-output 'out/*' \
#     --declared-output 'out/*/*' --env SERVED_RUN=<serve.sh run id> --env HZ_MEM_GB=<GB> \
#     --env HZ_CACHE=/workspace/research/cache/pouw-hidden-zk --env HZ_JOB=benchmarks/pouw/served_zk/job.sh -- bash benchmarks/pouw/hidden_zk/scope.sh CPUS GB
# In order, each step's files under out/:
#   1. hidden_zk/job.sh's builds (cached under $HZ_CACHE): flock-circuit (proving and selftest builds), Lean flock-verify, a venv
#   2. commit: served_commit.py commit on the "request" pass (an o_proj call picked from os.urandom, or SZ_CALL), then on the
#      "other" pass at the same call index: <p>/commitment.json (public), the prover's rows and salts in scratch
#   3. register and receive, one-stage's (verity/registration/v1, one root per side, window.kind "served"): <p>/registration.json,
#      the verifier's R1-R6 (<p>/check.json) and its receipt (<p>/receipt.json), before any draw
#   4. draw: Lean flock-verify draw --population N --k 1, from the verifier's own randomness, after the receipt (draw.json)
#   5. stage: stage_tile.py on each pass's rows at the drawn tile: Pc8TileHidden{K=4096,1,1,sm120}
#   6. the checks, each as expected, into summary.json:
#        receive-honest        the verifier receives the request's registration                             RECEIVED
#        receive-foreign       the request's commitment's ports, against the other request's registration    REFUSED
#        open-honest           the request's proof opens the drawn tile of the request's registration        ACCEPT
#        open-foreign-proof    the other request's proof, against the request's registration                 REJECT
#        open-foreign-commit   the request's proof, against the other request's registration                 REJECT
#        lean-honest-own       the Lean verifier on the request's --zk session and public file               ACCEPT
#        lean-foreign-pub      the same session against the other request's public file                      REJECT
#        lean-st-<case>        the selftest build's records (verity/flock-circuit+seed-injection): honest ACCEPT;
#                              message_byte_differs_from_the_committed_row (a tampered byte of an opened row),
#                              output_row_committed_false, unit_internal_bit_flipped REJECT
#        audit-<case>          one-stage's audit record (verity/one-stage/audit/v0, audit-<case>.json): the registration,
#                              receipt, draw and statement with the Lean verdict of record, the opening check and Rust's.
#                              honest ACCEPT; foreign-proof, foreign-commit and st-<each tampered case> REJECT
#      The Rust side's verdicts: the --zk session's own (prove.out) and each selftest case's (NEG lines).
set -uo pipefail
SRC=$(pwd)
Z=$SRC/benchmarks/pouw/served_zk   # not J: hidden_zk/job.sh sets J to its own directory
export HZ_CONFIGS=served HZ_N=1 HZ_MEM_GB=${HZ_MEM_GB:-96} HZ_RUNS=${HZ_RUNS:-1} HZ_WARM=${HZ_WARM:-0}
export HZ_CASES=${HZ_CASES:-honest message_byte_differs_from_the_committed_row output_row_committed_false unit_internal_bit_flipped}
# shellcheck source=../hidden_zk/job.sh
source "$SRC/benchmarks/pouw/hidden_zk/job.sh"
WORK=${WORK:-/workspace/pouw/mvp-e2e}
PASSES=$WORK/passes/${SERVED_RUN:?serve.sh run id}
export PYTHONPATH=$SRC:$SRC/catalog:$SRC/kernels:$SRC/experimental:$SRC/integrations/vllm:$SRC/benchmarks/pouw:$SRC/backends/flock/python
SC="$Z/served_commit.py"

setup
log "served-zk: passes $PASSES"
for p in request other; do [ -f "$PASSES/$p/manifest.json" ] || { log "no $PASSES/$p/manifest.json"; exit 2; }; done

declare -A VERDICT EXPECT
expect() { EXPECT[$1]=$2; VERDICT[$1]=$3; log "  $1: $3 (expected $2)"; }
mkdir -p "$OUT/request" "$OUT/other"
CALL=${SZ_CALL:-}
for p in request other; do
  log "commit: $p${CALL:+, call $CALL}"
  $TIME "$OUT/commit-$p.time" "$WORK/venv312/bin/python" "$SC" commit "$PASSES/$p" "$W/$p" ${CALL:+--call "$CALL"} \
    > "$OUT/commit-$p.out" 2>&1 || { log "commit FAILED: $(tail -3 "$OUT/commit-$p.out")"; exit 3; }
  cp "$W/$p/commitment.json" "$OUT/$p/"
  CALL=$("$PY" -c "import json; print(json.load(open('$OUT/$p/commitment.json'))['binding']['matmul'])")
  log "commit: $p call $CALL: $(tail -1 "$OUT/commit-$p.out")"
  log "register: $p: $("$PY" "$SC" register "$OUT/$p/commitment.json" "$OUT/$p/registration.json")"
done
log "receive (the verifier's R1-R6 and its receipt, before the draw)"
v=$("$PY" "$SC" receive "$OUT/request/commitment.json" "$OUT/request/registration.json" "$OUT/request" | tee "$OUT/receive-honest.out" | cut -d' ' -f1)
expect receive-honest RECEIVED "${v:-ERROR}"
"$PY" "$SC" receive "$OUT/other/commitment.json" "$OUT/other/registration.json" "$OUT/other" > "$OUT/receive-other.out"
v=$("$PY" "$SC" receive "$OUT/request/commitment.json" "$OUT/other/registration.json" "$W/receive-foreign" | tee "$OUT/receive-foreign.out" | cut -d' ' -f1)
expect receive-foreign REFUSED "${v:-ERROR}"
[ -f "$OUT/request/receipt.json" ] && [ -f "$OUT/other/receipt.json" ] || { log "receive FAILED: $(cat "$OUT/receive-honest.out" "$OUT/receive-other.out")"; exit 3; }
POP=$("$PY" -c "import json; print(json.load(open('$OUT/request/registration.json'))['population'])")
log "draw: $LEAN draw --population $POP --k 1 (the verifier's own randomness, after the receipt)"
"$LEAN" draw --population "$POP" --k 1 > "$OUT/draw.out" 2>&1 || { log "draw FAILED: $(tail -2 "$OUT/draw.out")"; exit 3; }
sed -n 's/^DRAW //p' "$OUT/draw.out" > "$OUT/draw.json"
log "draw: $(cat "$OUT/draw.json")"

D=$W/served-k4096 O=$OUT/served-k4096 DO=$W/other-k4096
mkdir -p "$D" "$O" "$DO"
for x in "request $D" "other $DO"; do
  read -r p dir <<< "$x"
  log "stage: $p"
  $TIME "$OUT/stage-$p.time" bash -c "$(vlimit "$STAGE_GB") exec $PY $Z/stage_tile.py $W/$p $OUT/draw.json $dir" > "$OUT/stage-$p.out" 2>&1 \
    || { log "stage FAILED: $(tail -3 "$OUT/stage-$p.out")"; exit 4; }
  cp "$dir/stage.json" "$OUT/$p/stage.json"; cp "$dir/opening.json" "$dir/pin.txt" "$OUT/$p/"; cp "$dir/pub.bin" "$OUT/$p/pub.bin"
done
log "staged: $("$PY" -c "import json; s = json.load(open('$OUT/request/stage.json')); print(s['definition'], s['ands'], 'ANDs, k_log', s['k_log'], 'units', s['draw']['units'], s['seconds'], 's')")"

opencheck() {   # opencheck TAG EXPECT SIDE(registration and commitment) PUB OPENING
  local v; v=$("$PY" "$SC" check "$OUT/$3/registration.json" "$OUT/$3/commitment.json" "$OUT/draw.json" "$4" "$5" --out "$OUT/$1.json" \
    | tee "$OUT/$1.out" | cut -d' ' -f1)
  expect "$1" "$2" "${v:-ERROR}"
}
log "opening checks"
opencheck open-honest ACCEPT request "$D/pub.bin" "$D/opening.json"
opencheck open-foreign-proof REJECT request "$DO/pub.bin" "$D/opening.json"
opencheck open-foreign-commit REJECT other "$D/pub.bin" "$DO/opening.json"

log "--zk session ($WARM warm + $RUNS timed)"
session "$D" "$O" || log "--zk session FAILED: $(tail -2 "$O/prove.err")"
controls "$D" "$O" hidden
lean_jobs "$D" "$O" hidden
s=$(ls -d "$D"/sessions/*/ 2>/dev/null | tail -1)
if [ -n "$s" ]; then
  $TIME "$O/lean-foreign-pub.time" bash -c "$(vlimit "$STAGE_GB") exec $LEAN verify --zk --circuit $D/circuit.txt \
    --statement verity/flock-circuit --public $DO/pub.bin --session $s" > "$O/lean-foreign-pub.out" 2> "$O/lean-foreign-pub.err"
fi
lean_verdict() { "$PY" -c "import json, sys; v = [json.loads(l[8:]) for l in open(sys.argv[1]) if l.startswith('VERDICT ')]; \
print(('ACCEPT' if v[0]['accepted'] else 'REJECT') if v else 'NONE')" "$1" 2>/dev/null || echo NONE; }
expect lean-honest-own ACCEPT "$(lean_verdict "$O/lean-honest-own.out")"
expect lean-foreign-pub REJECT "$(lean_verdict "$O/lean-foreign-pub.out")"
for c in $HZ_CASES; do
  [ -f "$O/lean-st-$c.out" ] || continue
  expect "lean-st-$c" "$([ "$c" = honest ] && echo ACCEPT || echo REJECT)" "$(lean_verdict "$O/lean-st-$c.out")"
done

auditrec() {   # auditrec CASE EXPECT SIDE(registration and receipt) STAGE PUB LEAN OPENING [RUST]
  local v; v=$("$PY" "$SC" audit "$OUT/$3/registration.json" "$OUT/$3/receipt.json" "$OUT/draw.json" "$4" "$5" "$OUT/audit-$1.json" \
    --lean "$6" --opening "$7" ${8:+--rust "$8"} | tee "$OUT/audit-$1.out" | cut -d' ' -f2)
  expect "audit-$1" "$2" "${v:-ERROR}"
}
log "audit records (verity/one-stage/audit/v0)"
auditrec honest ACCEPT request "$OUT/request/stage.json" "$D/pub.bin" "$O/lean-honest-own.out" "$OUT/open-honest.json" "$O/prove.out"
auditrec foreign-proof REJECT request "$OUT/other/stage.json" "$DO/pub.bin" "$O/lean-foreign-pub.out" "$OUT/open-foreign-proof.json"
auditrec foreign-commit REJECT other "$OUT/request/stage.json" "$D/pub.bin" "$O/lean-honest-own.out" "$OUT/open-foreign-commit.json" "$O/prove.out"
for c in $HZ_CASES; do
  [ -f "$O/lean-st-$c.out" ] || continue
  auditrec "st-$c" "$([ "$c" = honest ] && echo ACCEPT || echo REJECT)" request "$OUT/request/stage.json" "$D/pub.bin" \
    "$O/lean-st-$c.out" "$OUT/open-honest.json"
done

{
  echo "{"
  echo "\"served_run\": \"$SERVED_RUN\", \"call\": $CALL,"
  echo "\"stage\": $(cat "$OUT/request/stage.json"),"
  echo "\"rust\": {\"live\": $("$PY" -c "import json, sys; print(json.dumps([json.loads(l[5:]).get('verdict') for l in open(sys.argv[1]) if l.startswith('LIVE\t')]))" "$O/prove.out" 2>/dev/null || echo null),"
  echo "  \"selftest\": {$(for c in $HZ_CASES; do printf '"%s": %s,' "$c" "$("$PY" -c "import json, sys; print(json.dumps([l for l in open(sys.argv[1]) if l.startswith(('NEG', 'POS', 'VERDICT', 'OK'))][-3:]))" "$O/selftest-$c.out" 2>/dev/null || echo null)"; done | sed 's/,$//')}},"
  echo "\"checks\": {$(for t in "${!VERDICT[@]}"; do printf '"%s": {"verdict": "%s", "expected": "%s"},' "$t" "${VERDICT[$t]}" "${EXPECT[$t]}"; done | sed 's/,$//')}"
  echo "}"
} > "$OUT/summary.json"
ok=1
for t in "${!VERDICT[@]}"; do [ "${VERDICT[$t]}" = "${EXPECT[$t]}" ] || ok=0; done
log "served-zk: $([ $ok = 1 ] && echo 'every check as expected' || echo 'a check NOT as expected'): $("$PY" -c "import json; print(json.dumps({k: v['verdict'] for k, v in json.load(open('$OUT/summary.json'))['checks'].items()}))")"
"$PY" -c "import json; json.load(open('$OUT/summary.json'))" || log "summary.json is not JSON"
[ $ok = 1 ]
