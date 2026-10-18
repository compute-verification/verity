#!/usr/bin/env bash
# V*'s algebra (lane rec-v0): rec-algebra's InnerRepCheck (GfResiduals_v1{S}) on each rep of an inner session (80-rec-inner.sh's
# out/inner, with its claims.json) staged as one C-Flock statement (verity_flock.rec_residuals: w the prover's private row, v
# a value the verifier registers itself, the residuals registered by the verifier as zero; no public input), proved with --zk
# at a full draw (both instances; serve/prove on loopback, the server's keyed coins), and verified by Lean's flock-verify
# verify --zk under the verifier's program, partition and registered reads, and by the live build's replay --zk. Then the same circuit
# with the verifier's v at a forged inner transcript (rep 0's first challenge coin flipped: rec_inner.py forge --what
# challenge) beside the honest w, which both verifiers must reject. Run from the verity tree (research run --cwd source).
# Outputs under out/: host.txt, stage.json, time-stage.txt, honest/ and forged/ (time-prove.txt, prove.txt/.err, serve.log,
# lean.log, time-lean.txt, replay.txt, sizes.txt), summary.json.
# Lean checks each statement's v against the verifier's own coins (verify --rec-inner-*, RecCoins.check): the inner statement
# REC_STMT and INNER's session, the one the verifier served, for the forged statement too.
# env: INNER=DIR (required: claims.json, session/)  CPUS=32  FLOCK_WORK (as 80-rec-inner.sh)  REC_STMT=DIR (the inner
#      statement: circuit.txt, pub-N.bin and the verity/partition/v1 object it binds, partition.json; default INNER/stage)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
if [ -z "${RI_PINNED:-}" ]; then
  . verity/ml/flock/pod/cpu-slices.sh
  hold_slices "" "${CPUS:-32}" || exit 1
  export RI_PINNED=$SLICE_CPUS RAYON_NUM_THREADS=$(echo $SLICE_CPUS | tr ',' '\n' | wc -l)
  exec taskset -c $SLICE_CPUS bash "$0" "$@"
fi
[ -f "${INNER:?an inner session directory}/claims.json" ] || { echo "INNER=$INNER: no claims.json" >&2; exit 1; }
. verity/ml/flock/pod/rec-env.sh

FI=$S/forged-inner; rm -rf $FI; mkdir -p $FI && cp -r $INNER/session $INNER/public.json $INNER/private.json $INNER/claims.json $FI/
python3 verity/ml/flock/pod/rec_inner.py forge --what challenge --inner $FI > $O/forged.json || exit 1
/usr/bin/time -v -o $O/time-stage.txt python3 -m verity_flock.rec_residuals $INNER $S/st --forged $FI > $O/stage.json 2> $O/stage.err
echo "stage exit $?"
wait $LP
REC_STMT=${REC_STMT:-$INNER/stage}; REC_PUB=($REC_STMT/pub-*.bin)
REC_INNER=(--rec-inner-circuit $REC_STMT/circuit.txt --rec-inner-public ${REC_PUB[0]} --rec-inner-partition $REC_STMT/partition.json
  --rec-inner-session $INNER/session)

free_port() { python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }

# prove CASE: serve --zk, prove --zk, then Lean and the live build's replay on the session
prove() {
  local C=$S/st/$1 D=$O/$1
  mkdir -p $D
  [ -f $C/inst.bin ] || { tail -20 $O/stage.err; return 1; }
  local PIN=$(sha512sum $C/circuit.txt | cut -d' ' -f1) PORT=$(free_port)
  $B serve --zk --listen 127.0.0.1:$PORT --out $D/sessions --instances $C/pub.bin --circuit $C/circuit.txt --pin $PIN \
    --operator loopback > $D/serve.log 2>&1 &
  local SP=$!
  for i in $(seq 3600); do grep -q SERVING $D/serve.log && break; kill -0 $SP 2>/dev/null || break; sleep 1; done
  /usr/bin/time -v -o $D/time-prove.txt $B prove --zk --verifier 127.0.0.1:$PORT --instances $C/inst.bin --circuit $C/circuit.txt \
    --warm 0 --runs 1 > $D/prove.txt 2> $D/prove.err
  echo "$1 prove exit $?"
  grep -h panicked $D/prove.err | head -3
  kill $SP; wait $SP 2>/dev/null
  local SESS=$(find $D/sessions -name session.json -printf '%h\n' | sort | head -1)
  [ -n "$SESS" ] || { echo "$1: no session"; return 1; }
  /usr/bin/time -v -o $D/time-lean.txt $LEAN_BIN verify --zk --statement verity/flock-circuit --circuit $C/circuit.txt --public $C/pub.bin \
    --partition $C/partition.json --program $C/program.json --registered $C/registered.json "${REC_INNER[@]}" --session $SESS > $D/lean.log 2>&1
  echo "$1 lean exit $?"
  $B replay --zk --instances $C/pub.bin --circuit $C/circuit.txt --sessions $SESS > $D/replay.txt 2>&1
  echo "$1 replay exit $?"
  cp $C/registered.json $C/partition.json $D/
  du -sb $C/inst.bin $C/pub.bin $C/circuit.txt > $D/sizes.txt
}

prove honest
prove forged

python3 - $O $INNER <<'EOF'
import json, re, sys
from pathlib import Path
o, inner = Path(sys.argv[1]), sys.argv[2]

def text(p):
    return p.read_text() if p.exists() else ""

def tv(p, key):
    m = re.search(key + r".*: (\S+)", text(p))
    return m.group(1) if m else None

def gb(p):
    rss = tv(p, r"Maximum resident set size \(kbytes\)")
    return int(rss) / 2**20 if rss else None

stage = json.loads(text(o / "stage.json").splitlines()[0]) if text(o / "stage.json") else None

def verdict(t):
    # the partition note quotes S, which can hold a newline
    at = t.find("VERDICT ")
    return json.JSONDecoder(strict=False).raw_decode(t[at + 8:])[0] if at >= 0 else None

def one(d):
    lean = verdict(text(d / "lean.log"))
    up = next((json.loads(l.split("\t", 1)[1]) for l in text(d / "replay.txt").splitlines() if l.startswith("UPSTREAM")), None)
    live = [json.loads(l[5:]) for l in text(d / "prove.txt").splitlines() if l.startswith("LIVE")]
    return {"stage": (stage or {}).get(d.name), "prove_wall": tv(d / "time-prove.txt", r"Elapsed \(wall clock\) time"),
            "prover_peak_gb": gb(d / "time-prove.txt"),
            "live": {k: live[0].get(k) for k in ("accepted", "prove_s", "prove_total_s", "e2e_s", "proof_bytes", "wire_sent", "wire_recv",
                                                 "k_log", "dense_m", "instances")} if live else None,
            "lean": {**{k: lean.get(k) for k in ("accepted", "why", "registered")},
                     "partition": {k: v for k, v in (lean.get("partition") or {}).items() if k != "note"}} if lean else text(d / "lean.log")[-400:],
            "lean_wall": tv(d / "time-lean.txt", r"Elapsed \(wall clock\) time"), "lean_peak_gb": gb(d / "time-lean.txt"),
            "replay": up if up else text(d / "replay.txt")[-400:], "sizes": text(d / "sizes.txt")}

s = {"inner": inner, "shape": (stage or {}).get("shape"), "build_s": (stage or {}).get("build_s"),
     "stage_wall": tv(o / "time-stage.txt", r"Elapsed \(wall clock\) time"), "stage_peak_gb": gb(o / "time-stage.txt"),
     "stage_err": None if stage else text(o / "stage.err")[-2000:], "forged_what": json.loads(text(o / "forged.json") or "null"),
     "honest": one(o / "honest"), "forged": one(o / "forged")}
(o / "summary.json").write_text(json.dumps(s, indent=1))
print(json.dumps(s))
EOF
true
