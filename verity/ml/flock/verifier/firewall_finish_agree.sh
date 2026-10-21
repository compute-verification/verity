#!/usr/bin/env bash
# The firewall's final round in Lean (firewall_finish_agree.py): one firewalled CPU `prove --zk` session (FC_FIREWALL=1
# FC_HOST_WITNESS=1, through the Lean session loop of this tree, whose finish reads each table's level 0 from the loop, draws
# each rep's salts for levels ≥ 1 and holds the opening's and the inner proof's rounds itself) that dumps each table's roots
# and requests and each rep's request and Lean proof file (FC_FINISH_DUMP), then `flock-firewall finish` on the requests;
# the session verifier must pass every rep, and accept the session but for a refusal of its statement form. Run from the
# source tree's root:
#   research run --on vy-nebius-1 --project verity --source . --cwd source --declared-output 'out/*' --cpus 16 --mem-gb 48 \
#     -- bash verity/ml/flock/verifier/firewall_finish_agree.sh
# The builds, the staging and their cache are benchmarks/pouw/hidden_zk/job.sh's (HZ_CACHE); the statement is one of its
# configs (FF_CONFIG, default tile:256) over FF_N instances (default 1), all drawn. Results in $RESEARCH_RUN_DIR/out:
# result.json, the session's logs, the requests' sizes; never a request or a proof.
set -uo pipefail
SRC=$(pwd)
CFG=${FF_CONFIG:-tile:256}
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
stage_config "$role" "$k" "$tag" "$D" "$O" || { log "$tag: staging FAILED: $(tail -1 "$O/stage.err")"; exit 1; }
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
# the proofs that left are the Lean finish's alone: the session's verifier must pass every rep and accept the session as it
# would were its statement's form one the soundness proof covers (`without_refusal`: a tile statement's form is refused)
"$PY" -c "import json, sys; v = [json.loads(l[5:]) for l in open(sys.argv[1]) if l.startswith('LIVE\t')]; \
w = [x.get('without_refusal') or x['verdict'].get('without_refusal') or x for x in v]; \
print('the session verifier:', [(x['accepted'], x['verdict'].get('why', '')[:120]) for x in v], 'without refusal:', [y['accepted'] for y in w]); \
sys.exit(0 if v and all(y['accepted'] and all(r['ok'] for r in x['verdict']['reps']) for x, y in zip(v, w)) else 1)" "$O/prove.out" \
  | tee -a "$OUT/job.log"
[ "${PIPESTATUS[0]}" = 0 ] || rc=1
exit $rc
