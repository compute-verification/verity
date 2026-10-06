#!/usr/bin/env bash
# served_zk's verifier side on node 2's CPUs: a hidden tile proof of one served call, against that call's commitment in the
# ZK statement's format, after serve.sh's run SERVED_RUN retained its two requests under $WORK/passes/$SERVED_RUN.
#   research run --on vy-nebius-2 --project verity --campaign pouw --source <tree> --cwd source --declared-output 'out/*' \
#     --declared-output 'out/*/*' --env SERVED_RUN=<serve.sh run id> --env HZ_MEM_GB=<GB> \
#     --env HZ_CACHE=/workspace/research/cache/pouw-hidden-zk --env HZ_JOB=benchmarks/pouw/served_zk/job.sh -- bash benchmarks/pouw/hidden_zk/scope.sh CPUS GB
# In order, each step's files under out/:
#   1. hidden_zk/job.sh's builds (cached under $HZ_CACHE): flock-circuit (proving and selftest builds), Lean flock-verify, a venv
#   2. commit: served_commit.py commit on the "request" pass (an o_proj call drawn from os.urandom), then on the "other" pass
#      at the same call index: request/commitment.json, other/commitment.json (public), the prover's rows and salts in scratch
#   3. draw: served_commit.py draw against request/commitment.json, after it is written (draw.json); the same source against
#      other/commitment.json (draw-other.json, the foreign control's)
#   4. stage: stage_tile.py on each pass's rows at the drawn tile: Pc8TileHidden{K=4096,1,1,sm120}
#   5. the checks, each ACCEPT or REJECT as expected, into summary.json:
#        open-honest          the request's proof opens the drawn rows of the request's commitment         ACCEPT
#        open-foreign-proof   the other request's proof, against the request's commitment                  REJECT
#        open-foreign-commit  the request's proof, against the other request's commitment                  REJECT
#        lean-honest-own      the Lean verifier on the request's --zk session and public file              ACCEPT
#        lean-foreign-pub     the same session against the other request's public file                     REJECT
#        lean-st-<case>       the selftest build's records (verity/flock-circuit+seed-injection): honest ACCEPT;
#                             message_byte_differs_from_the_committed_row (a tampered byte of an opened row),
#                             output_row_committed_false, unit_internal_bit_flipped REJECT
#      The Rust side's verdicts: the --zk session's own (prove.out) and each selftest case's (NEG lines).
set -uo pipefail
SRC=$(pwd)
J=$SRC/benchmarks/pouw/served_zk
export HZ_CONFIGS=served HZ_N=1 HZ_MEM_GB=${HZ_MEM_GB:-96} HZ_RUNS=${HZ_RUNS:-1} HZ_WARM=${HZ_WARM:-0}
export HZ_CASES=${HZ_CASES:-honest message_byte_differs_from_the_committed_row output_row_committed_false unit_internal_bit_flipped}
# shellcheck source=../hidden_zk/job.sh
source "$SRC/benchmarks/pouw/hidden_zk/job.sh"
WORK=${WORK:-/workspace/pouw/mvp-e2e}
PASSES=$WORK/passes/${SERVED_RUN:?serve.sh run id}
export PYTHONPATH=$SRC:$SRC/catalog:$SRC/kernels:$SRC/experimental:$SRC/integrations/vllm:$SRC/benchmarks/pouw:$SRC/backends/flock/python
SC="$J/served_commit.py"

setup
log "served-zk: passes $PASSES"
for p in request other; do [ -f "$PASSES/$p/manifest.json" ] || { log "no $PASSES/$p/manifest.json"; exit 2; }; done

log "commit: request"
$TIME "$OUT/commit-request.time" "$WORK/venv312/bin/python" "$SC" commit "$PASSES/request" "$W/request" > "$OUT/commit-request.out" 2>&1 \
  || { log "commit FAILED: $(tail -3 "$OUT/commit-request.out")"; exit 3; }
mkdir -p "$OUT/request" "$OUT/other" && cp "$W/request/commitment.json" "$OUT/request/"
CALL=$("$PY" -c "import json; print(json.load(open('$W/request/commitment.json'))['binding']['matmul'])")
log "commit: request call $CALL: $(tail -1 "$OUT/commit-request.out")"
log "draw (after the commitment is written)"
"$PY" "$SC" draw "$W/request/commitment.json" "$OUT/draw.json" | tee -a "$OUT/job.log"
log "commit: other, call $CALL"
$TIME "$OUT/commit-other.time" "$WORK/venv312/bin/python" "$SC" commit "$PASSES/other" "$W/other" --call "$CALL" > "$OUT/commit-other.out" 2>&1 \
  || { log "commit FAILED: $(tail -3 "$OUT/commit-other.out")"; exit 3; }
cp "$W/other/commitment.json" "$OUT/other/"
"$PY" "$SC" draw "$W/other/commitment.json" "$OUT/draw-other.json" \
  --source "$("$PY" -c "import json; print(json.load(open('$OUT/draw.json'))['source'])")" > /dev/null

D=$W/served-k4096 O=$OUT/served-k4096 DO=$W/other-k4096
mkdir -p "$D" "$O" "$DO"
for x in "request $D draw.json" "other $DO draw-other.json"; do
  read -r p dir dr <<< "$x"
  log "stage: $p"
  $TIME "$OUT/stage-$p.time" bash -c "$(vlimit "$STAGE_GB") exec $PY $J/stage_tile.py $W/$p $OUT/$dr $dir" > "$OUT/stage-$p.out" 2>&1 \
    || { log "stage FAILED: $(tail -3 "$OUT/stage-$p.out")"; exit 4; }
  cp "$dir/stage.json" "$OUT/$p/stage.json"; cp "$dir/opening.json" "$dir/pin.txt" "$OUT/$p/"; cp "$dir/pub.bin" "$OUT/$p/pub.bin"
done
log "staged: $("$PY" -c "import json; s = json.load(open('$OUT/request/stage.json')); print(s['definition'], s['ands'], 'ANDs, k_log', s['k_log'], 'draw', s['draw']['a'], s['draw']['b'], s['seconds'], 's')")"

declare -A VERDICT EXPECT
expect() { EXPECT[$1]=$2; VERDICT[$1]=$3; log "  $1: $3 (expected $2)"; }
opencheck() {   # opencheck TAG EXPECT COMMIT DRAW PUB OPENING
  local v; v=$("$PY" "$SC" check "$3" "$4" "$5" "$6" | tee "$OUT/$1.out" | cut -d' ' -f1)
  expect "$1" "$2" "${v:-ERROR}"
}
log "opening checks"
opencheck open-honest ACCEPT "$OUT/request/commitment.json" "$OUT/draw.json" "$D/pub.bin" "$D/opening.json"
opencheck open-foreign-proof REJECT "$OUT/request/commitment.json" "$OUT/draw.json" "$DO/pub.bin" "$D/opening.json"
opencheck open-foreign-commit REJECT "$OUT/other/commitment.json" "$OUT/draw-other.json" "$D/pub.bin" "$DO/opening.json"

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
