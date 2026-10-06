#!/usr/bin/env bash
# served_zk's window through the proof service on node 2's CPUs: SZ_CALLS served calls of a retained request committed as
# one audit window (padded with filler tiles to a grid level), registered with exactly PoUW's public items, received,
# drawn subset:K' once, every drawn tile proved hidden and opened, a filler tile proved beside them, and the outcome; after
# serve.sh's run SERVED_RUN retained its two requests under $WORK/passes/$SERVED_RUN.
#   research run --on vy-nebius-2 --project verity --campaign pouw --source <tree> --cwd source --declared-output 'out/*' \
#     --declared-output 'out/*/*' --env SERVED_RUN=<serve.sh run id> --env HZ_MEM_GB=<GB> \
#     --env HZ_CACHE=/workspace/research/cache/pouw-hidden-zk --env HZ_JOB=benchmarks/pouw/served_zk/job.sh -- bash benchmarks/pouw/hidden_zk/scope.sh CPUS GB
# In order, each step's files under out/:
#   1. hidden_zk/job.sh's builds (cached under $HZ_CACHE): flock-circuit (proving and selftest builds), Lean flock-verify, a venv
#   2. params: the verifier's own parameters of each request's window (<p>/params.json: its context, ε, δ, the grid, P, |W|)
#   3. commit: served_commit.py commit of the same SZ_CALLS o_proj calls (picked from os.urandom, or SZ_CALL_LIST) on the
#      "request" and the "other" pass: <p>/commitment.json (public), <p>/advice.json (the prover's entries), rows in scratch
#   4. register, receive, select through the service (service.py, P1's calls): <p>/registration.json, recv-<p>/{check,receipt}.json,
#      draw.json (Lean flock-verify draw --population N --k K', the verifier's own randomness, after the receipt, once)
#   5. per drawn unit u<t>: stage_tile.py (Pc8TileHidden{K=4096,1,1,sm120}), the --zk session, the opening check, Lean;
#      the selftest build's controls on the first; a filler unit's the same, and the other request's tile staged
#   6. outcome: one-stage's audit record over every drawn unit and the outcome's public items (outcome-<case>/)
#   7. the checks, each as expected, into summary.json:
#        register-extra        a registration with one more public item (the calls' shapes)                    REFUSED
#        receive-honest        the verifier receives the request's registration                                RECEIVED
#        receive-foreign       the other request's registration, against the request's parameters              REFUSED
#        receive-extra         the request's registration with one more public key                             REFUSED
#        receive-raised        the request's registration claiming the next grid level (more credit)           REFUSED
#        select                Lean's draw of K' units, after the receipt                                     DRAWN
#        select-again          a second select of the same receipt: no retry redraws                           REFUSED
#        open-u<t>             each drawn unit's proof opens that unit's rows under the window root            ACCEPT
#        lean-u<t>             the Lean verifier on each drawn unit's --zk session                             ACCEPT
#        open-filler           the filler unit's proof opens its zero rows under the window root              ACCEPT
#        lean-filler           the Lean verifier on the filler unit's --zk session                             ACCEPT
#        shape-filler          the filler's proof and a drawn real tile's: circuit, class, sizes               SAME
#        open-foreign-proof    the other request's proof, against the request's registration                   REJECT
#        open-foreign-commit   the request's proof, against the other request's registration                   REJECT
#        lean-foreign-pub      the first unit's session against the other request's public file               REJECT
#        lean-st-<case>        the selftest build's records on the first unit: honest ACCEPT; the tampered REJECT
#        outcome-honest        every drawn unit's verdicts, one audit record; public = PoUW's items            ACCEPT
#        outcome-again         a second outcome of the same registration                                        REFUSED
#        outcome-foreign-proof the first unit's proof, opening and Lean verdict the other request's             REJECT
#        outcome-st-<case>     the first unit's Lean verdict a selftest case's: honest ACCEPT; the tampered    REJECT
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
VPY=$WORK/venv312/bin/python

setup
log "served-zk window: passes $PASSES"
for p in request other; do [ -f "$PASSES/$p/manifest.json" ] || { log "no $PASSES/$p/manifest.json"; exit 2; }; done

declare -A VERDICT EXPECT
expect() { EXPECT[$1]=$2; VERDICT[$1]=$3; log "  $1: $3 (expected $2)"; }
first() { cut -d' ' -f1; }
jq_() { "$PY" -c "import json, sys; d = json.load(open(sys.argv[1])); print(eval(sys.argv[2], {'d': d, 'json': json}))" "$@"; }
mkdir -p "$OUT/request" "$OUT/other"

# ---- 2-3. params and commit ---------------------------------------------------------------------------------------------------
for p in request other; do
  "$VPY" "$SC" params "$PASSES/$p" "$OUT/$p/params.json" > "$OUT/params-$p.out" 2>&1 || { log "params FAILED: $(tail -3 "$OUT/params-$p.out")"; exit 3; }
done
CALLS=${SZ_CALL_LIST:-$("$PY" -c "import json, random, sys; m = json.load(open(sys.argv[1])); \
c = sorted(u['index'] for u in m['matmuls'] if u['k'] == u['n'] == 4096); print(' '.join(map(str, sorted(random.SystemRandom().sample(c, int(sys.argv[2]))))))" \
  "$PASSES/request/manifest.json" "${SZ_CALLS:-2}")}
log "window: calls $CALLS (os.urandom), params $(cat "$OUT/request/params.json")"
CALLARGS=$(for c in $CALLS; do printf -- '--call %s ' "$c"; done)
for p in request other; do
  # shellcheck disable=SC2086
  $TIME "$OUT/commit-$p.time" "$VPY" "$SC" commit "$PASSES/$p" "$OUT/$p/params.json" "$W/$p" $CALLARGS > "$OUT/commit-$p.out" 2>&1 &
done
wait
for p in request other; do
  [ -f "$W/$p/commitment.json" ] || { log "commit $p FAILED: $(tail -3 "$OUT/commit-$p.out")"; exit 3; }
  cp "$W/$p/commitment.json" "$W/$p/advice.json" "$OUT/$p/"
  log "commit: $p: $(tail -1 "$OUT/commit-$p.out")"
done

# ---- 4. register, receive, select ---------------------------------------------------------------------------------------------
for p in request other; do
  log "register: $p: $("$PY" "$SC" register "$W/$p" "$OUT/$p/params.json" "$OUT/$p/registration.json" 2>&1 | tail -1)"
  [ -f "$OUT/$p/registration.json" ] || { log "register $p FAILED"; exit 3; }
done
v=$("$PY" "$SC" register "$W/request" "$OUT/request/params.json" "$W/extra.json" --extra 'shapes=[[1,4096,4096]]' --expect-refused \
  | tee "$OUT/register-extra.out" | first)
expect register-extra REFUSED "${v:-ERROR}"
R=$OUT/request/registration.json RP=$OUT/request/params.json
v=$("$PY" "$SC" receive "$RP" "$R" "$OUT/recv-request" | tee "$OUT/receive-honest.out" | first)
expect receive-honest RECEIVED "${v:-ERROR}"
"$PY" "$SC" receive "$OUT/other/params.json" "$OUT/other/registration.json" "$OUT/recv-other" > "$OUT/receive-other.out"
v=$("$PY" "$SC" receive "$RP" "$OUT/other/registration.json" "$W/recv-foreign" | tee "$OUT/receive-foreign.out" | first)
expect receive-foreign REFUSED "${v:-ERROR}"
"$PY" "$SC" edit "$R" "$OUT/registration-extra.json" 'window.public.calls=2'
v=$("$PY" "$SC" receive "$RP" "$OUT/registration-extra.json" "$W/recv-extra" | tee "$OUT/receive-extra.out" | first)
expect receive-extra REFUSED "${v:-ERROR}"
"$PY" "$SC" edit "$R" "$OUT/registration-raised.json" "window.public.work.level=$(( $(jq_ "$R" "d['window']['public']['work']['level']") + 1 ))"
v=$("$PY" "$SC" receive "$RP" "$OUT/registration-raised.json" "$W/recv-raised" | tee "$OUT/receive-raised.out" | first)
expect receive-raised REFUSED "${v:-ERROR}"
[ -f "$OUT/recv-request/receipt.json" ] || { log "receive FAILED: $(cat "$OUT/receive-honest.out")"; exit 3; }
RC=$OUT/recv-request/receipt.json
log "select: $LEAN draw (the verifier's own randomness, after the receipt)"
v=$("$PY" "$SC" select "$RP" "$R" "$RC" "$OUT/draw.json" --lean "$LEAN" --log "$OUT/service" | tee "$OUT/select.out" | first)
expect select DRAWN "${v:-ERROR}"
v=$("$PY" "$SC" select "$RP" "$R" "$RC" "$W/draw-again.json" --lean "$LEAN" --log "$OUT/service" --expect-refused | tee "$OUT/select-again.out" | first)
expect select-again REFUSED "${v:-ERROR}"
[ -f "$OUT/draw.json" ] || { log "select FAILED: $(cat "$OUT/select.out")"; exit 3; }
UNITS=$(jq_ "$OUT/draw.json" "' '.join(map(str, sorted(d['units'])))")
FILL=$(jq_ "$OUT/request/advice.json" "[e for e in d['entries'] if e['kind'] == 'filler'][0]['lo']")
REAL=$(jq_ "$OUT/request/advice.json" "d['real']")
log "draw: units $UNITS of $(jq_ "$R" "d['population']") (real tiles $REAL, the filler from $FILL)"

# ---- 5. the units -------------------------------------------------------------------------------------------------------------
stage_unit() {   # stage_unit TAG SIDE UNIT: $W/TAG staged, its stage.json, opening.json, pin.txt and pub.bin copied to $OUT/TAG
  local tag=$1 p=$2 t=$3
  mkdir -p "$W/$tag" "$OUT/$tag"
  $TIME "$OUT/$tag/stage.time" bash -c "$(vlimit "$STAGE_GB") exec $PY $Z/stage_tile.py $W/$p $OUT/$p/params.json $t $W/$tag" \
    > "$OUT/$tag/stage.out" 2>&1 || { log "stage $tag FAILED: $(tail -3 "$OUT/$tag/stage.out")"; return 1; }
  cp "$W/$tag/stage.json" "$W/$tag/opening.json" "$W/$tag/pin.txt" "$W/$tag/pub.bin" "$OUT/$tag/"
  log "stage $tag (unit $t): $(jq_ "$OUT/$tag/stage.json" "(d['kind'], d['entry'], d['ands'], d['k_log'], d['seconds'])")"
}
opencheck() {    # opencheck TAG EXPECT SIDE UNIT PUB OPENING [--undrawn]
  local v; v=$("$PY" "$SC" check "$OUT/$3/params.json" "$OUT/$3/registration.json" "$OUT/draw.json" "$4" "$5" "$6" ${7:-} \
    --out "$OUT/$1.json" | tee "$OUT/$1.out" | first)
  expect "$1" "$2" "${v:-ERROR}"
}
prove_unit() {   # prove_unit TAG: the --zk session, then its Lean checks in the background
  session "$W/$1" "$OUT/$1" || log "$1: --zk session FAILED: $(tail -2 "$OUT/$1/prove.err")"
  ( lean_jobs "$W/$1" "$OUT/$1" hidden ) &
}
lean_verdict() { "$PY" -c "import json, sys; v = [json.loads(l[8:]) for l in open(sys.argv[1]) if l.startswith('VERDICT ')]; \
print(('ACCEPT' if v[0]['accepted'] else 'REJECT') if v else 'NONE')" "$1" 2>/dev/null || echo NONE; }

U0=
REAL_UNIT=
for t in $UNITS; do
  stage_unit "u$t" request "$t" || continue
  opencheck "open-u$t" ACCEPT request "$t" "$OUT/u$t/pub.bin" "$OUT/u$t/opening.json"
  if [ -z "$U0" ]; then U0=$t; controls "$W/u$t" "$OUT/u$t" hidden; fi
  [ "$t" -lt "$FILL" ] && [ -z "$REAL_UNIT" ] && REAL_UNIT=$t
  prove_unit "u$t"
done
[ -n "$U0" ] || { log "no drawn unit staged"; exit 4; }
if [ -z "$REAL_UNIT" ]; then stage_unit u0-real request 0 && prove_unit u0-real && REAL_UNIT=0-real; fi
DRAWN_FILLER=$(for t in $UNITS; do [ "$t" -ge "$FILL" ] && echo "$t"; done | head -1)
FU=${DRAWN_FILLER:-$FILL}
if [ -z "$DRAWN_FILLER" ]; then
  stage_unit filler request "$FU" && prove_unit filler
  opencheck open-filler ACCEPT request "$FU" "$OUT/filler/pub.bin" "$OUT/filler/opening.json" --undrawn
  FT=filler
else
  FT=u$FU
  expect open-filler ACCEPT "${VERDICT[open-u$FU]}"
fi
OU=$(( U0 < $(jq_ "$OUT/other/registration.json" "d['population']") ? U0 : 0 ))
stage_unit other-tile other "$OU"
opencheck open-foreign-proof REJECT request "$U0" "$OUT/other-tile/pub.bin" "$OUT/u$U0/opening.json"
opencheck open-foreign-commit REJECT other "$OU" "$OUT/u$U0/pub.bin" "$OUT/other-tile/opening.json" --undrawn
s=$(ls -d "$W/u$U0"/sessions/*/ 2>/dev/null | tail -1)
if [ -n "$s" ]; then
  ( $TIME "$OUT/u$U0/lean-foreign-pub.time" bash -c "$(vlimit "$STAGE_GB") exec $LEAN verify --zk --circuit $W/u$U0/circuit.txt \
      --statement verity/flock-circuit --public $OUT/other-tile/pub.bin --session $s" > "$OUT/u$U0/lean-foreign-pub.out" 2> "$OUT/u$U0/lean-foreign-pub.err" ) &
fi
log "waiting for the Lean checks"
wait
for t in $UNITS; do expect "lean-u$t" ACCEPT "$(lean_verdict "$OUT/u$t/lean-honest-own.out")"; done
expect lean-filler ACCEPT "$(lean_verdict "$OUT/$FT/lean-honest-own.out")"
expect lean-foreign-pub REJECT "$(lean_verdict "$OUT/u$U0/lean-foreign-pub.out")"
for c in $HZ_CASES; do
  [ -f "$OUT/u$U0/lean-st-$c.out" ] || continue
  expect "lean-st-$c" "$([ "$c" = honest ] && echo ACCEPT || echo REJECT)" "$(lean_verdict "$OUT/u$U0/lean-st-$c.out")"
done
v=$("$PY" "$SC" shape "$W/$FT" "$W/u$REAL_UNIT" --out "$OUT/shape-filler.json" | tee "$OUT/shape-filler.out" | first)
expect shape-filler SAME "${v:-ERROR}"

# ---- 6. outcome ---------------------------------------------------------------------------------------------------------------
units_json() {   # units_json OUT [FIRST_STAGE FIRST_PUB FIRST_LEAN FIRST_OPENING]: the drawn units' files, the first replaced
  "$PY" - "$OUT" "$1" "${2:-}" "${3:-}" "${4:-}" "${5:-}" $UNITS <<'PY'
import json, sys
out, dest, st, pub, lean, op, *units = sys.argv[1:]
rows = []
for i, t in enumerate(units):
    u = f"{out}/u{t}"
    row = {"unit": int(t), "stage": f"{u}/stage.json", "pub": f"{u}/pub.bin", "lean": f"{u}/lean-honest-own.out",
           "opening": f"{out}/open-u{t}.json", "rust": f"{u}/prove.out"}
    if i == 0:
        row.update({k: v for k, v in (("stage", st), ("pub", pub), ("lean", lean), ("opening", op)) if v})
        if lean:
            row.pop("rust")
    rows.append(row)
json.dump(rows, open(dest, "w"))
PY
}
outcome() {      # outcome CASE EXPECT UNITS_JSON [--expect-reject|--expect-refused]
  local v; v=$("$PY" "$SC" outcome "$RP" "$R" "$RC" "$OUT/draw.json" "$3" "$OUT/outcome-$1" --log "$W/olog-${5:-$1}" ${4:-} \
    | tee "$OUT/outcome-$1.out")
  case $v in OUTCOME*) v=$(echo "$v" | cut -d' ' -f2);; *) v=$(echo "$v" | first);; esac
  expect "outcome-$1" "$2" "${v:-ERROR}"
}
log "outcome (verity/one-stage/audit/v0 over every drawn unit)"
units_json "$OUT/units-honest.json"
outcome honest ACCEPT "$OUT/units-honest.json"
outcome again REFUSED "$OUT/units-honest.json" --expect-refused honest
U=$OUT/u$U0
units_json "$OUT/units-foreign-proof.json" "$OUT/other-tile/stage.json" "$OUT/other-tile/pub.bin" "$U/lean-foreign-pub.out" "$OUT/open-foreign-proof.json"
outcome foreign-proof REJECT "$OUT/units-foreign-proof.json" --expect-reject
for c in $HZ_CASES; do
  [ -f "$U/lean-st-$c.out" ] || continue
  units_json "$OUT/units-st-$c.json" "" "" "$U/lean-st-$c.out"
  if [ "$c" = honest ]; then outcome "st-$c" ACCEPT "$OUT/units-st-$c.json"; else outcome "st-$c" REJECT "$OUT/units-st-$c.json" --expect-reject; fi
done

# ---- 7. summary ---------------------------------------------------------------------------------------------------------------
{
  echo "{"
  echo "\"served_run\": \"$SERVED_RUN\", \"calls\": [$(echo "$CALLS" | tr ' ' ',')], \"units\": [$(echo "$UNITS" | tr ' ' ',')],"
  echo "\"real_tiles\": $REAL, \"filler_from\": $FILL, \"filler_unit\": $FU, \"params\": $(cat "$RP"),"
  echo "\"public\": $(jq_ "$OUT/outcome-honest/outcome.json" "json.dumps(d['public'])" 2>/dev/null || echo null),"
  echo "\"stage\": $(cat "$OUT/u$U0/stage.json"),"
  echo "\"shape\": $(cat "$OUT/shape-filler.json" 2>/dev/null || echo null),"
  echo "\"checks\": {$(for t in "${!VERDICT[@]}"; do printf '"%s": {"verdict": "%s", "expected": "%s"},' "$t" "${VERDICT[$t]}" "${EXPECT[$t]}"; done | sed 's/,$//')}"
  echo "}"
} > "$OUT/summary.json"
ok=1
for t in "${!VERDICT[@]}"; do [ "${VERDICT[$t]}" = "${EXPECT[$t]}" ] || ok=0; done
log "served-zk window: $([ $ok = 1 ] && echo 'every check as expected' || echo 'a check NOT as expected'): $("$PY" -c "import json; print(json.dumps({k: v['verdict'] for k, v in sorted(json.load(open('$OUT/summary.json'))['checks'].items())}))")"
"$PY" -c "import json; json.load(open('$OUT/summary.json'))" || log "summary.json is not JSON"
[ $ok = 1 ]
