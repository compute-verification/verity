#!/usr/bin/env bash
# served_zk's window through the proof system on a node's CPUs: SZ_CALLS served calls of a retained request committed as
# one audit window (padded with filler tiles to a grid level), registered with exactly PoUW's public items, received,
# drawn subset:K' once, every drawn tile proved hidden and opened, a filler tile proved beside them, and the outcome; after
# serve.sh's run SERVED_RUN retained its two requests under $WORK/passes/$SERVED_RUN.
#   research run --on MACHINE --project verity --source <tree> --cwd source --declared-output 'out/*' \
#     --declared-output 'out/*/*' --env SERVED_RUN=<serve.sh run id> --env HZ_MEM_GB=<GB> \
#     --env HZ_CACHE=/workspace/research/cache/pouw-hidden-zk --env HZ_JOB=benchmarks/pouw/served_zk/job.sh -- bash benchmarks/pouw/hidden_zk/scope.sh CPUS GB
# The drawn units run as a parallel fan-out across the scope's cores (step 5): SZ_JOBS workers at once (default HZ_MEM_GB * 3/4 /
# SZ_MEM_PER_JOB, capped to the CPUs; SZ_JOBS=1 is the old serial loop), SZ_MEM_PER_JOB the GB each is budgeted (default 13, the
# session's serve + prove peak). Each unit's verdicts go to its own file; a reduce after the fan-out folds them into the summary.
# In order, each step's files under out/:
#   1. hidden_zk/job.sh's builds (cached under $HZ_CACHE): flock-circuit (proving and selftest builds), Lean flock-verify, a venv
#   2. params: the verifier's own parameters of each request's window (<p>/params.json: its context, ε, δ, the grid, P, |W|)
#   3. commit: served_commit.py commit of the same SZ_CALLS o_proj calls (picked from os.urandom, or SZ_CALL_LIST) on the
#      "request" and the "other" pass: <p>/commitment.json (public), <p>/advice.json (the prover's entries), rows in scratch
#   4. register (the committer: <p>/registration.json), then one challenger process (served_commit.py challenger, over P1's
#      sampled_proofs.service Stream, holding the epoch's coins: challenger/coins.json, the commitment it publishes first;
#      its requests and replies in challenger/requests.jsonl, the coins it opened in challenger/openings.jsonl): its salt
#      coin opened after the request pass's weights' root (salt-opening.json), receive (Stream.register: receipt.json;
#      the challenger re-starts the pass's epoch from its manifest under its own commitment, the request pass, salted by a
#      beacon before epoch coins, as a recorded replay) and select (draw.json: the epoch's draw coin opened after the
#      registration, once, and Lean flock-verify draw --population N --k K' --stream over its key)
#   5. per drawn unit u<t>, as a parallel fan-out (SZ_JOBS at once): stage_tile.py (Pc8TileSide{K=4096,1,1,sm120}), the --zk
#      session, the opening check, Lean; the selftest build's controls on the first; a filler unit's the same, and the other request's tile staged
#   6. deliver (the drawn units' session records' SHA-512) and outcome (one-stage's audit record over every drawn unit and
#      PoUW's public items, outcome-<case>/); each control on a stream of its own that replays the honest draw
#   7. the checks, each as expected, into summary.json:
#        coins                 the challenger's coin commitment, published before any request                  COINS
#        coin-salt             its salt coin, opened after the request pass's weights' root                    OPENED
#        receive-coin-<case>   the request pass carrying a salt coin of another challenger (foreign), opened
#                              after what the verifier had not recorded (early), labelled the draw coin
#                              (mislabelled), a byte flipped (tampered), or the genuine coin beside the
#                              beacon's salt (resalted)                                                        REFUSED
#        receive-unrecorded    the request pass, salted by a beacon, not received as a recorded replay         REFUSED
#        register-extra        a registration with one more public item (the calls' shapes)                    REFUSED
#        receive-foreign       the other request's registration, against the request's parameters              REFUSED
#        receive-extra         the request's registration with one more public key                             REFUSED
#        receive-raised        the request's registration claiming the next grid level (more credit)           REFUSED
#        receive-honest        the challenger receives the request's registration                              REGISTERED
#        receive-open          a second registration while the first has no outcome                            REFUSED
#        select                Lean's draw of K' units over the draw coin's key, opened after the receipt    DRAWN
#        select-again          a second select of the same registration: no retry redraws                     REFUSED
#        receive-after-draw    the request's registration on a stream of its own, after the epoch's draw coin  REFUSED
#        open-u<t>             each drawn unit's proof opens that unit's rows under the window root            ACCEPT
#        lean-u<t>             the Lean verifier on each drawn unit's --zk session, every check passed and
#                              refused for the tile's want of a Program (no --program: NO_PROGRAM)             REFUSED
#        open-filler           the filler unit's proof opens its zero rows under the window root              ACCEPT
#        lean-filler           the Lean verifier on the filler unit's --zk session, as lean-u<t>               REFUSED
#        shape-filler          the filler's proof and a real tile's: circuit, class, proof files' sizes and
#                              the session record's skeleton (its integers, string lengths)                      SAME
#        open-foreign-proof    the other request's proof, against the request's registration                   REJECT
#        open-foreign-commit   the request's proof, against the other request's registration                   REJECT
#        lean-foreign-pub      the first unit's session against the other request's public file               REJECT
#        lean-st-<case>        the selftest build's records on the first unit: honest REFUSED; the tampered REJECT
#        deliver-honest        the drawn units' session records, once                                           DELIVERED
#        outcome-honest        every drawn unit's verdicts, one audit record; public = PoUW's items: the Lean
#                              verifier accepts no unit without its Program                                    REJECT
#        outcome-again         a second outcome of the same registration: final                                 REFUSED
#        deliver-again         a second delivery of the same draw                                               REFUSED
#        outcome-foreign-proof the first unit's proof, opening and Lean verdict the other request's             REJECT
#        retry-foreign-proof   a control's registration on that control's stream after its reject (budget 1)   REFUSED
#        outcome-st-<case>     the first unit's Lean verdict a selftest case's: every case                     REJECT
#        replay                the challenger's requests again from its recorded coins, Lean drawing          SAME
#        replay-twin           the same, stream_subset (Lean's draw's Python twin) drawing                    SAME
set -uo pipefail
SRC=$(pwd)
Z=$SRC/benchmarks/pouw/served_zk   # not J: hidden_zk/job.sh sets J to its own directory
export HZ_CONFIGS=served HZ_N=1 HZ_MEM_GB=${HZ_MEM_GB:-96} HZ_RUNS=${HZ_RUNS:-1} HZ_WARM=${HZ_WARM:-0}
export HZ_CASES=${HZ_CASES:-honest message_byte_differs_from_the_committed_row output_row_committed_false unit_internal_bit_flipped}
# shellcheck source=../hidden_zk/job.sh
source "$SRC/benchmarks/pouw/hidden_zk/job.sh"
WORK=${WORK:-/workspace/pouw/mvp-e2e}
PASSES=$WORK/passes/${SERVED_RUN:?serve.sh run id}
export PYTHONPATH=$SRC:$SRC/tools/config:$SRC/integrations/vllm:$SRC/benchmarks/pouw:$SRC/verity/ml/flock/python:$SRC/verity/core/difftest/python
SC="$Z/served_commit.py"
VPY=${SZ_PYTHON:-$WORK/venv312/bin/python}

setup
[ -x "$VPY" ] || VPY=$PY
log "served-zk window: passes $PASSES"
for p in request other; do [ -f "$PASSES/$p/manifest.json" ] || { log "no $PASSES/$p/manifest.json"; exit 2; }; done

declare -A VERDICT EXPECT
expect() { EXPECT[$1]=$2; VERDICT[$1]=$3; log "  $1: $3 (expected $2)"; }
first() { cut -d' ' -f1; }
jq_() { "$PY" -c "import json, sys; d = json.load(open(sys.argv[1])); print(eval(sys.argv[2], {'d': d, 'json': json}))" "$@"; }
mkdir -p "$OUT/request" "$OUT/other"
# the Lean checks of record (of_record.Record), built here so a failure stops the job before the challenger asks for them
command -v lake >/dev/null || export PATH="$HOME/.elan/bin:$PATH"
{ (cd "$SRC/verity/core/protocols/pouw/lean/check" && lake build pouw-check pouw-check-difftest) \
  && (cd "$SRC/verity/core/service/sampled_proofs" && lake build sampled-proofs-difftest); } > "$OUT/lean-record-build.log" 2>&1 \
  || { log "Lean of record build FAILED: $(tail -5 "$OUT/lean-record-build.log")"; exit 3; }
log "Lean of record: pouw-check, pouw-check-difftest, sampled-proofs-difftest built"

# ---- 2-3. params and commit ---------------------------------------------------------------------------------------------------
# SZ_EPS / SZ_DELTA drive the draw at a non-default (ε, δ) for a realistic-parameter audit (k_prime grows with ln(1/δ)/ε);
# unset, served_commit's own defaults stand, so the default run is unchanged.
PARG=()
[ -n "${SZ_EPS:-}" ] && PARG+=(--eps "$SZ_EPS")
[ -n "${SZ_DELTA:-}" ] && PARG+=(--delta "$SZ_DELTA")
for p in request other; do
  "$VPY" "$SC" params "$PASSES/$p" "$OUT/$p/params.json" "${PARG[@]}" > "$OUT/params-$p.out" 2>&1 || { log "params FAILED: $(tail -3 "$OUT/params-$p.out")"; exit 3; }
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
"$PY" "$SC" edit "$R" "$OUT/registration-extra.json" 'window.public.calls=2'
"$PY" "$SC" edit "$R" "$OUT/registration-raised.json" "window.public.work.level=$(( $(jq_ "$R" "d['window']['public']['work']['level']") + 1 ))"
# the challenger: one process holding P1's streams and the epoch's coins from the receipt to the outcomes; ask() is its one channel
# A pass salted by epoch coins (manifest v2) was salted by the serving controller's challenger, whose entropy this one holds
MAN=$PASSES/request/manifest.json
V1=$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['schema'] == 'verity-vllm/pouw-pearl-c-run/v1')" "$MAN")
STATE=${SZ_COIN_STATE:-$WORK/coin-state/$SERVED_RUN}
if [ "$V1" = True ]; then REC=true STATEARG=(); else REC=false STATEARG=(--state "$STATE"); fi
log "pass: $("$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['schema'])" "$MAN"), the challenger ${STATEARG[*]:-fresh}"
coproc SVC { exec "$PY" "$SC" challenger "$RP" "$OUT/challenger" --lean "$LEAN" "${STATEARG[@]}" 2> "$OUT/challenger.err"; }
ask() {          # ask CHECK JSON: one request; its reply line in $OUT/CHECK.out, its word in $REPLY
  local line=
  if [ -n "${SVC[1]:-}" ]; then echo "$2" >&"${SVC[1]}" && IFS= read -r line <&"${SVC[0]}"; fi
  echo "${line:-ERROR the challenger is gone: $(tail -2 "$OUT/challenger.err" 2>/dev/null)}" > "$OUT/$1.out"
  REPLY=$(first < "$OUT/$1.out")
}
req() { "$PY" -c "import json, sys; a = sys.argv[1:]; print(json.dumps(dict(zip(a[::2], a[1::2]))))" "$@"; }
ask coins "$(req op coins)"
expect coins COINS "$REPLY"
if [ "$V1" != True ]; then
  v=$("$PY" -c "import json, sys; print('SAME' if json.load(open(sys.argv[1])) == json.load(open(sys.argv[2])) else 'DIFFERENT')" \
    "$OUT/challenger/coins.json" "$PASSES/coins.json" 2>/dev/null)
  expect coins-pass SAME "${v:-ERROR}"
fi
WR=$("$PY" -c "import json, sys; from verity.core.protocols.pouw.audit import weights_root; m = json.load(open(sys.argv[1])); \
print(weights_root({int(w): bytes.fromhex(v['root']) for w, v in m['weights'].items()}).hex())" "$MAN")
ask coin-salt "$(req op salt after "$WR" out "$OUT/salt-opening.json")"
expect coin-salt OPENED "$REPLY"
for c in foreign early mislabelled tampered resalted; do
  "$PY" "$SC" coin-variant "$MAN" "$OUT/challenger/coins.json" "$OUT/salt-opening.json" "$W/coin-$c.json" "$c" || log "coin-variant $c FAILED"
  ask "receive-coin-$c" "$(req op register name "coin-$c" registration "$R" manifest "$W/coin-$c.json")"
  expect "receive-coin-$c" REFUSED "$REPLY"
done
UNREC=$MAN
if [ "$V1" != True ]; then   # the pass as one recorded before epoch coins would carry it: a beacon's salt, no coin
  UNREC=$W/beacon.json
  "$PY" -c "import json, os, sys; m = json.load(open(sys.argv[1])); [m.pop(k, None) for k in ('coins', 'salt_opening')]; \
m.update(schema='verity-vllm/pouw-pearl-c-run/v1', beacon=os.urandom(32).hex()); json.dump(m, open(sys.argv[2], 'w'))" "$MAN" "$UNREC"
fi
ask receive-unrecorded "$(req op register name unrecorded registration "$R" manifest "$UNREC")"
expect receive-unrecorded REFUSED "$REPLY"
ask receive-foreign "$(req op register name foreign registration "$OUT/other/registration.json" manifest "$PASSES/other/manifest.json" recorded $REC)"
expect receive-foreign REFUSED "$REPLY"
ask receive-extra "$(req op register name extra registration "$OUT/registration-extra.json" manifest "$MAN" recorded $REC)"
expect receive-extra REFUSED "$REPLY"
ask receive-raised "$(req op register name raised registration "$OUT/registration-raised.json" manifest "$MAN" recorded $REC)"
expect receive-raised REFUSED "$REPLY"
ask receive-honest "$(req op register name honest registration "$R" manifest "$MAN" recorded $REC out "$OUT/receipt.json")"
expect receive-honest REGISTERED "$REPLY"
ask receive-open "$(req op register name open registration "$R" manifest "$MAN" recorded $REC)"
expect receive-open REFUSED "$REPLY"
[ -f "$OUT/receipt.json" ] || { log "receive FAILED: $(cat "$OUT/receive-honest.out")"; exit 3; }
log "select: the draw coin, opened after the registration, and $LEAN draw over its key"
ask select "$(req op select name honest out "$OUT/draw.json")"
expect select DRAWN "$REPLY"
ask select-again "$(req op select name honest out "$W/draw-again.json")"
expect select-again REFUSED "$REPLY"
ask receive-after-draw "$(req op register name after registration "$R" stream after manifest "$MAN" recorded $REC)"
expect receive-after-draw REFUSED "$REPLY"
[ -f "$OUT/draw.json" ] || { log "select FAILED: $(cat "$OUT/select.out")"; exit 3; }
UNITS=$(jq_ "$OUT/draw.json" "' '.join(map(str, sorted(d['units'])))")
FILL=$(jq_ "$OUT/request/advice.json" "[e for e in d['entries'] if e['kind'] == 'filler'][0]['lo']")
REAL=$(jq_ "$OUT/request/advice.json" "d['real']")
log "draw: units $UNITS of $(jq_ "$R" "d['population']") (real tiles $REAL, the filler from $FILL)"

# ---- 7. the challenger's replay and the summary -----------------------------------------------------------------------------
finish() {       # the challenger quits, then its requests again from its recorded coins, Lean drawing and the reference's
  [ -n "${SVC[1]:-}" ] && echo '{"op": "quit"}' >&"${SVC[1]}"
  [ -n "${SVC_PID:-}" ] && wait "$SVC_PID" 2>/dev/null
  local v
  v=$("$PY" "$SC" replay "$RP" "$OUT/challenger" "$W/replay" --lean "$LEAN" | tee "$OUT/replay.out" | first)
  expect replay SAME "${v:-ERROR}"
  v=$("$PY" "$SC" replay "$RP" "$OUT/challenger" "$W/replay-twin" | tee "$OUT/replay-twin.out" | first)
  expect replay-twin SAME "${v:-ERROR}"
}
summary() {      # summary.json
  {
    echo "{"
    echo "\"served_run\": \"$SERVED_RUN\", \"calls\": [$(echo "$CALLS" | tr ' ' ',')], \"units\": [$(echo "$UNITS" | tr ' ' ',')],"
    echo "\"real_tiles\": $REAL, \"filler_from\": $FILL, \"filler_unit\": ${FU:-null}, \"params\": $(cat "$RP"),"
    echo "\"public\": $(jq_ "$OUT/outcome-honest/outcome.json" "json.dumps(d['public'])" 2>/dev/null || echo null),"
    echo "\"coins\": $(jq_ "$OUT/outcome-honest/outcome.json" "json.dumps(d['coins'])" 2>/dev/null || echo null),"
    echo "\"stage\": $(cat "$OUT/u$U0/stage.json" 2>/dev/null || echo null),"
    echo "\"shape\": $(cat "$OUT/shape-filler.json" 2>/dev/null || echo null),"
    echo "\"checks\": {$(for t in "${!VERDICT[@]}"; do printf '"%s": {"verdict": "%s", "expected": "%s"},' "$t" "${VERDICT[$t]}" "${EXPECT[$t]}"; done | sed 's/,$//')}"
    echo "}"
  } > "$OUT/summary.json"
  local ok=1
  for t in "${!VERDICT[@]}"; do [ "${VERDICT[$t]}" = "${EXPECT[$t]}" ] || ok=0; done
  log "served-zk window: $([ $ok = 1 ] && echo 'every check as expected' || echo 'a check NOT as expected'): $("$PY" -c "import json; print(json.dumps({k: v['verdict'] for k, v in sorted(json.load(open('$OUT/summary.json'))['checks'].items())}))")"
  "$PY" -c "import json; json.load(open('$OUT/summary.json'))" || { log "summary.json is not JSON"; ok=0; }
  [ $ok = 1 ] || exit 1
}

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
  BG+=($!)
}
# REFUSED: rejected for the want of the verifier's program alone, once every check has passed (ProvedScope.NO_PROGRAM)
lean_verdict() { "$PY" -c "import json, sys; v = [json.loads(l[8:]) for l in open(sys.argv[1]) if l.startswith('VERDICT ')]; \
print(('ACCEPT' if v[0]['accepted'] else 'REFUSED' if v[0]['why'].startswith(sys.argv[2]) else 'REJECT') if v else 'NONE')" \
  "$1" "refused: a session without the verifier's program (--program)" 2>/dev/null || echo NONE; }

unit_work() {    # unit_work TAG SIDE UNIT: one unit's stage -> open -> (controls on FIRST) -> --zk session -> Lean, its
  local tag=$1 p=$2 t=$3 v vf                             # open-TAG and lean-TAG verdicts to its own file for the reduce below
  vf=$OUT/$tag/verdicts.tsv                               # a separate assignment: set -u expands a single local's words before any take effect
  mkdir -p "$OUT/$tag"; : > "$vf"
  stage_unit "$tag" "$p" "$t" || { printf 'open-%s\tACCEPT\tERROR\nlean-%s\tREFUSED\tNONE\n' "$tag" "$tag" > "$vf"; return 0; }
  v=$("$PY" "$SC" check "$OUT/$p/params.json" "$OUT/$p/registration.json" "$OUT/draw.json" "$t" "$OUT/$tag/pub.bin" "$OUT/$tag/opening.json" \
    --out "$OUT/open-$tag.json" | tee "$OUT/open-$tag.out" | first)
  printf 'open-%s\tACCEPT\t%s\n' "$tag" "${v:-ERROR}" > "$vf"
  [ "$t" = "$FIRST" ] && controls "$W/$tag" "$OUT/$tag" hidden
  session "$W/$tag" "$OUT/$tag" || log "$tag: --zk session FAILED: $(tail -2 "$OUT/$tag/prove.err")"
  lean_jobs "$W/$tag" "$OUT/$tag" hidden
  printf 'lean-%s\tREFUSED\t%s\n' "$tag" "$(lean_verdict "$OUT/$tag/lean-honest-own.out")" >> "$vf"
  rm -f "$W/$tag/inst.bin"; [ "$t" = "$FIRST" ] || rm -f "$W/$tag/circuit.txt"   # free the ~486 MB circuit; FIRST's stays for lean-foreign-pub
}

# A parallel fan-out over the drawn units: the session (serve + prove, ~12.5 GB together) bounds the worker count against the
# scope's memory; the stage and Lean (the ~60 s, ~2.5 GB bottleneck) are single-threaded, so throughput is in the count, not the
# threads. SZ_JOBS overrides the count (SZ_JOBS=1 is the old serial loop), SZ_MEM_PER_JOB the GB each is budgeted.
JOBS=${SZ_JOBS:-$(( MEM_GB * 3 / 4 / ${SZ_MEM_PER_JOB:-13} ))}
[ "$JOBS" -lt 1 ] && JOBS=1
[ "$JOBS" -gt "$TH" ] && JOBS=$TH
WT=$(( TH / JOBS )); [ "$WT" -lt 1 ] && WT=1
FIRST=${UNITS%% *}                                        # the unit the controls run on; the resolved U0 (first that staged) below
REAL_UNIT=; for t in $UNITS; do [ "$t" -lt "$FILL" ] && { REAL_UNIT=$t; break; }; done
log "units: $(echo "$UNITS" | wc -w) drawn, $JOBS workers x $WT threads ($MEM_GB GB, $TH CPUs, ${SZ_MEM_PER_JOB:-13} GB each), controls on u$FIRST"
PIDS=()
for t in $UNITS; do
  ( export RAYON_NUM_THREADS=$WT OMP_NUM_THREADS=$WT OPENBLAS_NUM_THREADS=$WT; unit_work "u$t" request "$t" ) &
  PIDS+=($!)
  [ "${#PIDS[@]}" -ge "$JOBS" ] && { wait "${PIDS[0]}" 2>/dev/null; PIDS=("${PIDS[@]:1}"); }
done
for p in "${PIDS[@]}"; do wait "$p" 2>/dev/null; done
log "units: fan-out done, reducing per-unit verdicts"
for t in $UNITS; do
  vf=$OUT/u$t/verdicts.tsv
  if [ -f "$vf" ]; then while IFS=$'\t' read -r k e g; do [ -n "$k" ] && expect "$k" "$e" "$g"; done < "$vf"
  else expect "open-u$t" ACCEPT ERROR; expect "lean-u$t" REFUSED NONE; fi
done
U0=; for t in $UNITS; do [ -f "$OUT/u$t/stage.json" ] && { U0=$t; break; }; done
[ -n "$U0" ] || { log "no drawn unit staged: $(tail -1 "$OUT/u$FIRST/stage.out" 2>/dev/null)"; finish; summary; exit 3; }
if [ "$U0" != "$FIRST" ]; then   # the pre-chosen unit's stage failed: put the controls (and circuit for lean-foreign-pub) on the resolved U0
  [ -f "$W/u$U0/circuit.txt" ] || stage_unit "u$U0" request "$U0"
  [ -d "$W/u$U0/st" ] || { controls "$W/u$U0" "$OUT/u$U0" hidden; lean_jobs "$W/u$U0" "$OUT/u$U0" hidden; }
fi

BG=()
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
  BG+=($!)
fi
log "waiting for the post-loop Lean checks"
[ "${#BG[@]}" -gt 0 ] && wait "${BG[@]}"   # not the challenger: the filler, u0-real and foreign-pub Lean (the drawn units' ran in the fan-out)
expect lean-filler REFUSED "$(lean_verdict "$OUT/$FT/lean-honest-own.out")"
expect lean-foreign-pub REJECT "$(lean_verdict "$OUT/u$U0/lean-foreign-pub.out")"
for c in $HZ_CASES; do
  [ -f "$OUT/u$U0/lean-st-$c.out" ] || continue
  expect "lean-st-$c" "$([ "$c" = honest ] && echo REFUSED || echo REJECT)" "$(lean_verdict "$OUT/u$U0/lean-st-$c.out")"
done
for d in "$FT" "u$REAL_UNIT"; do
  s=$(ls -d "$W/$d"/sessions/*/ 2>/dev/null | tail -1); [ -n "$s" ] && cp "$s/session.json" "$OUT/$d/session.json"
done
v=$("$PY" "$SC" shape "$W/$FT" "$W/u$REAL_UNIT" --out "$OUT/shape-filler.json" | tee "$OUT/shape-filler.out" | first)
expect shape-filler SAME "${v:-ERROR}"

# ---- 6. deliver and outcome ---------------------------------------------------------------------------------------------------
units_json() {   # units_json DEST [FIRST_STAGE FIRST_PUB FIRST_LEAN FIRST_OPENING FIRST_SESSION]: the drawn units' files, the first replaced
  "$PY" - "$OUT" "$W" "$1" "${2:-}" "${3:-}" "${4:-}" "${5:-}" "${6:-}" $UNITS <<'PY'
import json, sys
from pathlib import Path
out, w, dest, st, pub, lean, op, session, *units = sys.argv[1:]
rows = []
for i, t in enumerate(units):
    u = f"{out}/u{t}"
    sessions = sorted(Path(f"{w}/u{t}/sessions").glob("*/session.json"))
    row = {"unit": int(t), "stage": f"{u}/stage.json", "pub": f"{u}/pub.bin", "lean": f"{u}/lean-honest-own.out",
           "opening": f"{out}/open-u{t}.json", "rust": f"{u}/prove.out",
           "session": str(sessions[-1]) if sessions else f"{w}/u{t}/sessions/none"}
    if i == 0:
        row.update({k: v for k, v in (("stage", st), ("pub", pub), ("lean", lean), ("opening", op), ("session", session)) if v})
        if lean:
            row.pop("rust")
    rows.append(row)
json.dump(rows, open(dest, "w"))
PY
}
control() {      # control CASE EXPECT UNITS_JSON: a stream of its own replaying the honest draw, delivered and judged
  local n=ctl-$1
  ask "register-$1" "$(req op register name "$n" registration "$R" stream "$n" replay honest)"
  ask "select-$1" "$(req op select name "$n")"
  ask "deliver-$1" "$(req op deliver name "$n" units "$3")"
  ask "outcome-$1" "$(req op outcome name "$n" units "$3" out "$OUT/outcome-$1")"
  expect "outcome-$1" "$2" "$REPLY"
}
log "deliver and outcome (verity/one-stage/audit/v0 over every drawn unit)"
units_json "$OUT/units-honest.json"
ask deliver-honest "$(req op deliver name honest units "$OUT/units-honest.json")"
expect deliver-honest DELIVERED "$REPLY"
ask outcome-honest "$(req op outcome name honest units "$OUT/units-honest.json" out "$OUT/outcome-honest")"
expect outcome-honest REJECT "$REPLY"
ask outcome-again "$(req op outcome name honest units "$OUT/units-honest.json" out "$W/outcome-again")"
expect outcome-again REFUSED "$REPLY"
ask deliver-again "$(req op deliver name honest units "$OUT/units-honest.json")"
expect deliver-again REFUSED "$REPLY"
U=$OUT/u$U0
units_json "$OUT/units-foreign-proof.json" "$OUT/other-tile/stage.json" "$OUT/other-tile/pub.bin" "$U/lean-foreign-pub.out" "$OUT/open-foreign-proof.json"
control foreign-proof REJECT "$OUT/units-foreign-proof.json"
ask retry-foreign-proof "$(req op register name retry registration "$R" stream ctl-foreign-proof replay honest)"
expect retry-foreign-proof REFUSED "$REPLY"
for c in $HZ_CASES; do
  [ -f "$U/lean-st-$c.out" ] || continue
  units_json "$OUT/units-st-$c.json" "" "" "$U/lean-st-$c.out" "" "$W/u$U0/st/$c/session.json"
  control "st-$c" REJECT "$OUT/units-st-$c.json"
done
finish
summary
