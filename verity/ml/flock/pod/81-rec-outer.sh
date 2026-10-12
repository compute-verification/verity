#!/usr/bin/env bash
# V*'s outer side (lane rec-v0): each Ligerito level of an inner session (80-rec-inner.sh's out/inner) staged as one C-Flock
# statement of RecOpen instances (verity_flock.rec_outer: each opening's climb to the cap node its coin names, the expected
# rows registered by the verifier), proved with --zk at a full draw (every instance; serve/prove on loopback, the server's
# keyed coins), and verified by Lean's flock-verify verify --zk under the verifier's program, partition and registered reads,
# and by the live build's replay --zk. Then a forged inner transcript (rep 0's first query coin of level NEG_LEVEL flipped,
# so the verifier's expected row for that opening moves) staged and proved the same way, which both verifiers must reject.
# Run from the verity tree (research run --cwd source). Outputs under out/: host.txt, L<l>/ and forged-L<l>/ (stage.json,
# time-stage.txt, time-prove.txt, prove.txt/.err, serve.log, lean.log, replay.txt), summary.json.
# Lean checks each statement's coefficients and positions against the verifier's own coins (verify --rec-inner-*,
# RecCoins.check): the inner statement REC_STMT and INNER's session, the one the verifier served, for the forged level too.
# env: INNER=DIR (required: public.json, private.json, session/)  LEVELS=0,1,2,3 (default: every level)  NEG_LEVEL (default:
#      the last)  LINK=1 (the opened rows read from the prover's registration of them: rec_outer --link-rows)  CPUS=32
#      FLOCK_WORK (as 80-rec-inner.sh)  REC_STMT=DIR (the inner statement: circuit.txt, pub-N.bin and the verity/partition/v1
#      object it binds, partition.json; default INNER/stage)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
if [ -z "${RI_PINNED:-}" ]; then
  . verity/ml/flock/pod/cpu-slices.sh
  hold_slices "" "${CPUS:-32}" || exit 1
  export RI_PINNED=$SLICE_CPUS RAYON_NUM_THREADS=$(echo $SLICE_CPUS | tr ',' '\n' | wc -l)
  exec taskset -c $SLICE_CPUS bash "$0" "$@"
fi
[ -f "${INNER:?an inner session directory}/public.json" ] || { echo "INNER=$INNER: no public.json" >&2; exit 1; }
. verity/ml/flock/pod/rec-env.sh
wait $LP

ALL=$(python3 -c "
import sys; from pathlib import Path; from verity_flock import rec_outer as O, rec_vstar as V
pub, priv, proofs = V.load(Path(sys.argv[1])); print(','.join(str(l['level']) for l in O.levels(V.queries(pub, priv, proofs))))" $INNER)
LEVELS=${LEVELS:-$ALL}; NEG_LEVEL=${NEG_LEVEL:-${ALL##*,}}
LINK_ROWS=; [ "${LINK:-0}" = 1 ] && LINK_ROWS=--link-rows
REC_STMT=${REC_STMT:-$INNER/stage}; REC_PUB=($REC_STMT/pub-*.bin)
REC_INNER=(--rec-inner-circuit $REC_STMT/circuit.txt --rec-inner-public ${REC_PUB[0]} --rec-inner-partition $REC_STMT/partition.json
  --rec-inner-session $INNER/session)

free_port() { python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }

# outer INNER LEVEL NAME: stage, serve --zk, prove --zk, then Lean and the live build's replay on the session
outer() {
  local I=$1 l=$2 D=$O/$3 st=$S/stage-$3
  mkdir -p $D
  /usr/bin/time -v -o $D/time-stage.txt python3 -m verity_flock.rec_outer $I $st --levels $l $LINK_ROWS > $D/stage.json 2> $D/stage.err
  echo "$3 stage exit $?"
  local C=$st/L$l
  [ -f $C/inst.bin ] || { tail -20 $D/stage.err; return 1; }
  local PIN=$(sha512sum $C/circuit.txt | cut -d' ' -f1) PORT=$(free_port)
  $B serve --zk --listen 127.0.0.1:$PORT --out $D/sessions --instances $C/pub.bin --circuit $C/circuit.txt --pin $PIN \
    --operator loopback > $D/serve.log 2>&1 &
  local SP=$!
  for i in $(seq 1800); do grep -q SERVING $D/serve.log && break; kill -0 $SP 2>/dev/null || break; sleep 1; done
  /usr/bin/time -v -o $D/time-prove.txt $B prove --zk --verifier 127.0.0.1:$PORT --instances $C/inst.bin --circuit $C/circuit.txt \
    --warm 0 --runs 1 > $D/prove.txt 2> $D/prove.err
  echo "$3 prove exit $?"
  grep -h panicked $D/prove.err | head -3
  kill $SP; wait $SP 2>/dev/null
  local SESS=$(find $D/sessions -name session.json -printf '%h\n' | sort | head -1)
  [ -n "$SESS" ] || { echo "$3: no session"; return 1; }
  /usr/bin/time -v -o $D/time-lean.txt $LEAN_BIN verify --zk --statement verity/flock-circuit --circuit $C/circuit.txt --public $C/pub.bin \
    --partition $C/partition.json --program $C/program.json --registered $C/registered.json "${REC_INNER[@]}" --session $SESS > $D/lean.log 2>&1
  echo "$3 lean exit $?"
  $B replay --zk --instances $C/pub.bin --circuit $C/circuit.txt --sessions $SESS > $D/replay.txt 2>&1
  echo "$3 replay exit $?"
  cp $C/registered.json $C/partition.json $D/
  du -sb $C/inst.bin $C/pub.bin $C/circuit.txt > $D/sizes.txt
}

for l in ${LEVELS//,/ }; do outer $INNER $l L$l; done
FI=$S/forged-inner; rm -rf $FI; cp -r $INNER $FI
python3 verity/ml/flock/pod/rec_inner.py forge --what query --level $NEG_LEVEL --inner $FI > $O/forged.json
outer $FI $NEG_LEVEL forged-L$NEG_LEVEL

python3 - $O $INNER "$LEVELS" $NEG_LEVEL <<'EOF'
import json, re, sys
from pathlib import Path
o, inner, levels, neg = Path(sys.argv[1]), sys.argv[2], sys.argv[3].split(","), sys.argv[4]

def text(p):
    return p.read_text() if p.exists() else ""

def tv(p, key):
    m = re.search(key + r".*: (\S+)", text(p))
    return m.group(1) if m else None

def one(d):
    lean = next((json.loads(l.split(" ", 1)[1]) for l in text(d / "lean.log").splitlines() if l.startswith("VERDICT ")), None)
    up = next((json.loads(l.split("\t", 1)[1]) for l in text(d / "replay.txt").splitlines() if l.startswith("UPSTREAM")), None)
    live = [json.loads(l[5:]) for l in text(d / "prove.txt").splitlines() if l.startswith("LIVE")]
    rss = tv(d / "time-prove.txt", r"Maximum resident set size \(kbytes\)")
    return {"stage": json.loads(text(d / "stage.json").splitlines()[0]) if text(d / "stage.json") else None,
            "stage_wall": tv(d / "time-stage.txt", r"Elapsed \(wall clock\) time"),
            "prove_wall": tv(d / "time-prove.txt", r"Elapsed \(wall clock\) time"),
            "prover_peak_gb": int(rss) / 2**20 if rss else None,
            "live": {k: live[0].get(k) for k in ("accepted", "prove_s", "prove_total_s", "e2e_s", "proof_bytes", "wire_sent", "wire_recv",
                                                 "k_log", "dense_m", "instances")} if live else None,
            "lean": {k: lean.get(k) for k in ("accepted", "why", "partition", "registered")} if lean else text(d / "lean.log")[-400:],
            "lean_wall": tv(d / "time-lean.txt", r"Elapsed \(wall clock\) time"),
            "replay": up if up else text(d / "replay.txt")[-400:]}

s = {"inner": inner, "levels": {f"L{l}": one(o / f"L{l}") for l in levels},
     "forged": {"what": json.loads(text(o / "forged.json") or "null"), **one(o / f"forged-L{neg}")}}
(o / "summary.json").write_text(json.dumps(s, indent=1))
print(json.dumps(s))
EOF
true
