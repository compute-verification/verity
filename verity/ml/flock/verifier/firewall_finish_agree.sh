#!/usr/bin/env bash
# The firewall's final round in Lean (firewall_finish_agree.py): one firewalled CPU `prove --zk` session (FC_FIREWALL=1
# FC_HOST_WITNESS=1, through the Lean session loop of this tree, whose finish reads each table's level 0 from the loop, draws
# each rep's salts for levels ≥ 1 and holds the opening's and the inner proof's rounds itself) that dumps each table's roots
# and requests and each rep's request and Lean proof file (FC_FINISH_DUMP), then `flock-firewall finish` on the requests;
# the session's verifier must accept, every rep passing, and the Lean verifier (`flock-verify verify --zk`) pass every check
# (refused only for the tile's want of a Program). Run from the source tree's root:
#   bash verity/ml/flock/verifier/firewall_finish_agree.sh
#   (a job run from the source tree, 16 CPUs, 48 GB, keeping out/*)
# Every round in a device round's place is the loop's own (`Flock.Firewall.Frames`: `Skeleton.rounds`, then `Drive.openStep`).
# The builds, the staging and their cache are benchmarks/pouw/hidden_zk/job.sh's (HZ_CACHE); the statement is one of its
# configs (FF_CONFIG, default hidden:256: a hiding tile, whose rows are a form the soundness proof covers, where a plain
# tile's are refused) over FF_N instances (default 1), all drawn, or h1:K, the H1 tile unit at contraction K with one
# random instance (benchmarks/verifier_cost/flock_read.py), whose verifiers' verdicts are reported and not required: its
# checks of record are h1_prove's, with the window's Program, partition and registered reads. Results in
# $RESEARCH_RUN_DIR/out: result.json, the session's logs, the requests' sizes; never a request or a proof.
set -uo pipefail
SRC=$(pwd)
CFG=${FF_CONFIG:-hidden:256}
export HZ_CONFIGS=$CFG HZ_N=${FF_N:-1} HZ_MEM_GB=${HZ_MEM_GB:-48}
export HZ_CACHE=${HZ_CACHE:-/workspace/research/cache/pouw-hidden-zk}
# shellcheck source=../../../../benchmarks/pouw/hidden_zk/job.sh
source "$SRC/benchmarks/pouw/hidden_zk/job.sh"
setup
FIREWALL=${LEAN%/flock-verify}/flock-firewall
[ -x "$FIREWALL" ] || { log "missing: $FIREWALL"; exit 1; }
role=${CFG%%:*} k=${CFG#*:} tag=${CFG%%:*}-k${CFG#*:}
D=$W/$tag O=$OUT/$tag
mkdir -p "$D" "$O"
if [ "$role" = h1 ]; then
  "$PY" "$SRC/benchmarks/verifier_cost/flock_read.py" "$k" "$D/h1" > "$O/stage.out" 2> "$O/stage.err" \
    || { log "$tag: staging FAILED: $(tail -1 "$O/stage.err")"; exit 1; }
  ln -s "$D/h1/circuit.txt" "$D/h1/stmt/inst.bin" "$D/h1/stmt/pub.bin" "$D/"
  cp "$D/h1/build.json" "$D/stage.json"
  "$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['pin'], end='')" "$D/stage.json" > "$D/pin.txt"
else
  stage_config "$role" "$k" "$tag" "$D" "$O" || { log "$tag: staging FAILED: $(tail -1 "$O/stage.err")"; exit 1; }
fi
cp "$D/stage.json" "$O/"
DUMP=$W/dump; mkdir -p "$DUMP"
pin=$(cat "$D/pin.txt") port=$(free_port)
"$BIN" serve --zk --listen 127.0.0.1:$port --out "$D/sessions" --instances "$D/pub.bin" --circuit "$D/circuit.txt" --pin "$pin" \
  --operator loopback --draw "subset:$N" > "$O/serve.log" 2>&1 &
srv=$!
until grep -q '^SERVING ' "$O/serve.log" 2>/dev/null; do kill -0 $srv 2>/dev/null || { log "serve: $(tail -1 "$O/serve.log")"; exit 1; }; sleep 1; done
log "$tag: one firewalled session, dumping finish's requests"
FC_FIREWALL=1 FC_HOST_WITNESS=1 FC_FIREWALL_LEAN=$FIREWALL FC_FINISH_DUMP=$DUMP \
  $TIME "$O/prove.time" "$BIN" prove --zk --verifier 127.0.0.1:$port --instances "$D/inst.bin" --circuit "$D/circuit.txt" \
  --runs 1 --warm 0 > "$O/prove.out" 2> "$O/prove.err"
rc=$?
kill $srv 2>/dev/null; wait $srv 2>/dev/null
grep -h '^FIREWALL' "$O/prove.err" | head -5 | tee -a "$OUT/job.log"
[ $rc = 0 ] || { log "prove exit $rc: $(tail -3 "$O/prove.err")"; exit 1; }
J0=$(find "$D/sessions" -name session.json | head -1)
[ -n "$J0" ] || { log "no session.json under $D/sessions"; exit 1; }
S=$(dirname "$J0")
cp "$J0" "$O/"
(cd "$DUMP" && ls -l) > "$O/dump.ls"
ARGS=(--circuit "$D/circuit.txt" --public "$D/pub.bin" --draw "$S/session.json")
[ -e "$D/partition.json" ] && ARGS+=(--partition "$D/partition.json")
log "$tag: flock-firewall finish on $(ls "$DUMP" | wc -l) dumped files"
$TIME "$O/finish.time" "$PY" "$SRC/verity/ml/flock/verifier/firewall_finish_agree.py" --lean "$FIREWALL" --dump "$DUMP" --out "$OUT" \
  -- "${ARGS[@]}" 2>&1 | tee -a "$OUT/job.log"
rc=${PIPESTATUS[0]}
log "firewall-finish-agree: exit $rc"
# the proofs that left are the Lean finish's alone: the session's verifier must pass every rep and accept, and so must the
# Lean verifier
"$PY" -c "import json, sys; v = [json.loads(l[5:]) for l in open(sys.argv[1]) if l.startswith('LIVE\t')]; \
print('the session verifier:', [(x['accepted'], x['verdict'].get('why', '')[:160], [r['ok'] for r in x['verdict']['reps']]) for x in v]); \
sys.exit(0 if v and all(x['accepted'] and x['verdict']['reps'] and all(r['ok'] for r in x['verdict']['reps']) for x in v) else 1)" "$O/prove.out" \
  | tee -a "$OUT/job.log"
live=${PIPESTATUS[0]}
VARGS=(verify --zk --circuit "$D/circuit.txt" --statement verity/flock-circuit --public "$D/pub.bin" --session "$S")
[ -e "$D/partition.json" ] && VARGS+=(--partition "$D/partition.json")
"$LEAN" "${VARGS[@]}" > "$O/lean-verify.out" 2> "$O/lean-verify.err"
# a hidden_zk tile is no Program's, so the Lean verifier refuses its session for want of one (ProvedScope.NO_PROGRAM) once
# every check has passed: that refusal alone passes here, as agree.py compares it
"$PY" -c "import json, sys; v = [json.loads(l[8:]) for l in open(sys.argv[1]) if l.startswith('VERDICT ')]; \
sys.exit(0 if v and (v[0]['accepted'] or v[0]['why'].startswith(sys.argv[2])) else 1)" "$O/lean-verify.out" \
  "refused: a session without the verifier's program (--program)"
lean=$?
log "the session's verdict: $([ "$live" = 0 ] && echo ACCEPT || echo "NOT accepted"); the Lean verifier's: \
$([ $lean = 0 ] && echo "every check passed" || echo "REJECT") $(grep '^VERDICT ' "$O/lean-verify.out" | cut -c1-300)"
[ "$role" = h1 ] || { [ "$live" = 0 ] && [ $lean = 0 ]; } || rc=1
exit $rc
