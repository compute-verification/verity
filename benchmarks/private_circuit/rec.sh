#!/usr/bin/env bash
# The recursion's inner side (80-rec-inner.sh) over the universal unit: each class in CLS staged as universal.py's statement
# (--stage-only --keep-stage: a random hidden netlist padded to the class, N instances), proved on the CPU by the unmodified
# prover (FC_COINS=os) against verity_flock.rec_live's proxy, then replayed by upstream's verifier, by V*'s hashing
# (rec_vstar) and by Lean's flock-verify (--coins os), and Lean's InnerClaims run for V*'s algebra (83-rec-alg.sh). The
# classes run in order and stop at the first that fails. Each class's out/G<G>-N<N_IN>-O<N_OUT>/ holds stage.json, the
# statement, time-stage.txt / time-prove.txt (wall and peak RSS), prove.txt/.err, replay.txt, vstar.json, lean.log and inner/
# (81-rec-outer.sh's and 83-rec-alg.sh's INNER); out/summary.jsonl is one line per class.
#   research run --on vy-nebius-cpu-1 --project verity --source . --cwd source --declared-output 'out/**' -- \
#     python3 tools/verity/check/slot.py -- bash benchmarks/private_circuit/rec.sh CLS="1024,64,32 4096,64,32" N=4
# env: CLS (space-separated G,N_IN,N_OUT)  N=4 (instances)  CPUS=32  FLOCK_WORK (as 80-rec-inner.sh)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
if [ -z "${RI_PINNED:-}" ]; then
  . verity/ml/flock/pod/cpu-slices.sh
  hold_slices "" "${CPUS:-32}" || exit 1
  export RI_PINNED=$SLICE_CPUS RAYON_NUM_THREADS=$(echo $SLICE_CPUS | tr ',' '\n' | wc -l)
  exec taskset -c $SLICE_CPUS bash "$0" "$@"
fi
export FC_COINS=os
. verity/ml/flock/pod/rec-env.sh
OUT=$O
N=${N:-4}

CSHA=$( (echo $LSHA; cat verity/ml/flock/tests/lean/InnerClaims.lean) | sha256sum | cut -c1-16)
CLAIMS_BIN=$L/claims-$CSHA/inner-claims
(
  export PATH=$HOME/.elan/bin:$PATH
  exec 9>$L/claims.lock; flock 9
  if [ ! -x $CLAIMS_BIN ]; then
    C=$L/claims-src; rm -rf $C && mkdir -p $C && cp -r verity/core/service/flock/. $C/ && cp verity/ml/flock/tests/lean/InnerClaims.lean $C/ \
      && printf '\n[[lean_exe]]\nname = "inner-claims"\nroot = "InnerClaims"\n' >> $C/lakefile.toml \
      && (cd $C && lake build inner-claims) && mkdir -p $L/claims-$CSHA && cp $C/.lake/build/bin/inner-claims $CLAIMS_BIN
  fi
) > $OUT/build-claims.log 2>&1 &
CLP=$!
wait $LP

one_class() {
  local c=$1 G NI NO
  IFS=, read -r G NI NO <<< "$c"
  local O=$OUT/G$G-N$NI-O$NO U=$S/u-$G-$NI-$NO
  mkdir -p $O
  local D=$U/G$G-N$NI-O$NO/stage
  /usr/bin/time -v -o $O/time-stage.txt python3 benchmarks/private_circuit/universal.py --cls "$c" --n $N --stage-only --keep-stage \
    --out $U > $O/stage.json 2> $O/stage.err || { tail -20 $O/stage.err; return 1; }
  [ -f $D/circuit.txt ] || { cat $O/stage.json | cut -c1-2000; return 1; }
  local TB=(); [ -d $D/tables ] && TB=(--tables $D/tables)
  $B statement --instances $D/pub-$N.bin --circuit $D/circuit.txt "${TB[@]}" > $O/statement.txt 2>&1 || { tail -5 $O/statement.txt; return 1; }
  python3 -m verity_flock.rec_live --schedule-for $O/statement.txt > $O/schedule.json || return 1
  python3 -m verity_flock.rec_live --listen 127.0.0.1:0 --out $O/inner --schedule $O/schedule.json > $O/proxy.log 2>&1 &
  local PP=$! ADDR=
  for i in $(seq 600); do ADDR=$(awk '/^SERVING/ {print $2}' $O/proxy.log); [ -n "$ADDR" ] && break; kill -0 $PP 2>/dev/null || break; sleep 0.5; done
  [ -n "$ADDR" ] || { cat $O/proxy.log; return 1; }
  /usr/bin/time -v -o $O/time-prove.txt $B prove --verifier $ADDR --instances $D/inst-$N.bin --circuit $D/circuit.txt "${TB[@]}" \
    --runs 1 --warm 0 > $O/prove.txt 2> $O/prove.err
  echo "prove exit $?"
  wait $PP; echo "proxy exit $?"
  local SESS=$O/inner/session
  [ -f $SESS/session.json ] || { tail -20 $O/proxy.log $O/prove.err; return 1; }
  $B replay --instances $D/pub-$N.bin --circuit $D/circuit.txt "${TB[@]}" --sessions $SESS > $O/replay.txt 2>&1
  python3 -m verity_flock.rec_vstar $O/inner > $O/vstar.json 2> $O/vstar.err; echo "vstar exit $?"
  $LEAN_BIN verify --statement verity/flock-circuit --circuit $D/circuit.txt --public $D/pub-$N.bin --coins os --session $SESS > $O/lean.log 2>&1
  echo "lean exit $?"
  mkdir -p $O/inner/stage && cp $D/circuit.txt $D/pub-$N.bin $O/inner/stage/
  wait $CLP
  /usr/bin/time -v -o $O/time-claims.txt $CLAIMS_BIN verity/flock-circuit $D/circuit.txt $D/pub-$N.bin $SESS os > $O/inner/claims.json 2> $O/claims.err
  echo "claims exit $?"
  rm -rf $U
  python3 - $O "$c" $N >> $OUT/summary.jsonl <<'EOF'
import json, re, sys
from pathlib import Path
o, c, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
def tv(f):
    t = (o / f).read_text() if (o / f).exists() else ""
    m, w = re.search(r"Maximum resident set size \(kbytes\): (\d+)", t), re.search(r"Elapsed \(wall clock\) time.*: (\S+)", t)
    return {"peak_gb": round(int(m.group(1)) / 2**20, 3) if m else None, "wall": w.group(1) if w else None}
up = next((json.loads(l.split("\t", 1)[1])["verdict"] for l in (o / "replay.txt").read_text().splitlines() if l.startswith("UPSTREAM")), None)
lean = next((l[:300] for l in (o / "lean.log").read_text().splitlines() if "VERDICT" in l or "accepted" in l), (o / "lean.log").read_text()[-300:])
live = [json.loads(l[5:]) for l in (o / "prove.txt").read_text().splitlines() if l.startswith("LIVE")]
st = [json.loads(l) for l in (o / "stage.json").read_text().splitlines() if l.startswith("{")]
row = st[0] if st else {}
s = {"class": c, "n": n, "unit": row.get("unit"), "statement": {k: (row.get("statement") or {}).get(k) for k in ("k_log", "vus_per_block", "n")},
     "stage": tv("time-stage.txt"), "prove": tv("time-prove.txt"), "claims": tv("time-claims.txt"),
     "live": {k: live[0].get(k) for k in ("prove_s", "prove_total_s", "proof_bytes", "k_log", "dense_m")} if live else None,
     "replay": up, "lean": lean, "vstar": (o / "vstar.json").read_text()[:300]}
print(json.dumps(s))
EOF
  tail -1 $OUT/summary.jsonl
}

for c in ${CLS:?CLS="G,N_IN,N_OUT ..."}; do
  one_class "$c" || { echo "class $c failed"; exit 1; }
done
