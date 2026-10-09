#!/usr/bin/env bash
# The recursion's inner side (80-rec-inner.sh) over the universal unit: each class in CLS staged as universal.py's statement
# (--stage-only --keep-stage: a random hidden netlist padded to the class, N instances), proved on the CPU by the unmodified
# prover (FC_COINS=os) against verity_flock.rec_live's proxy, then replayed by upstream's verifier, by V*'s hashing
# (rec_vstar) and by Lean's flock-verify (--coins os), and Lean's InnerClaims run for V*'s algebra (83-rec-alg.sh). The
# classes run in order and stop at the first that fails. Each class's out/G<G>-N<N_IN>-O<N_OUT>/ holds stage.json, the
# statement, time-stage.txt / time-prove.txt (wall and peak RSS), prove.txt/.err, replay.txt, vstar.json, lean.log and inner/
# (81-rec-outer.sh's and 83-rec-alg.sh's INNER); out/summary.jsonl is one line per class.
# OUTER=1: the inner staged with plain SHA-512 leaves (FLOCK_PLAIN_LEAVES=1, a recursive session's), then V*'s side as
# 85-rec-reprice.sh's vstage, oprove and lean steps run it, on the CPU prover: the fold points (rec_vstage points), the extra
# claims (InnerFold.lean), each level's RecOpen statement and the algebra (rec_vstage stage), each proved --zk on loopback and
# checked by upstream's replay --zk and Lean's verify --zk, under out/G../outer/<statement>/.
#   research run --on vy-nebius-cpu-1 --project verity --source . --cwd source --declared-output 'out/**' -- \
#     python3 tools/verity/check/slot.py -- bash benchmarks/private_circuit/rec.sh CLS="1024,64,32 4096,64,32" N=4
# env: CLS (space-separated G,N_IN,N_OUT)  N=4 (instances)  CPUS=32  OUTER=0  FLOCK_WORK (as 80-rec-inner.sh)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
if [ -z "${RI_PINNED:-}" ]; then
  . verity/ml/flock/pod/cpu-slices.sh
  hold_slices "" "${CPUS:-32}" || exit 1
  export RI_PINNED=$SLICE_CPUS RAYON_NUM_THREADS=$(echo $SLICE_CPUS | tr ',' '\n' | wc -l)
  exec taskset -c $SLICE_CPUS bash "$0" "$@"
fi
export FC_COINS=os
STMT=verity/flock-circuit
[ "${OUTER:-0}" = 1 ] && export FLOCK_PLAIN_LEAVES=1 && STMT=verity/flock-circuit+plain-leaves
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
FSHA=$( (echo $LSHA; cat verity/ml/flock/tests/lean/InnerFold.lean) | sha256sum | cut -c1-16)
FOLD_BIN=$L/fold-$FSHA/inner-fold
FLP=
if [ "${OUTER:-0}" = 1 ]; then
  (
    export PATH=$HOME/.elan/bin:$PATH
    exec 9>$L/fold.lock; flock 9
    if [ ! -x $FOLD_BIN ]; then
      C=$L/fold-src; rm -rf $C && mkdir -p $C && cp -r verity/core/service/flock/. $C/ && cp verity/ml/flock/tests/lean/InnerFold.lean $C/ \
        && printf '\n[[lean_exe]]\nname = "inner-fold"\nroot = "InnerFold"\n' >> $C/lakefile.toml \
        && (cd $C && lake build inner-fold) && mkdir -p $L/fold-$FSHA && cp $C/.lake/build/bin/inner-fold $FOLD_BIN
    fi
  ) > $OUT/build-fold.log 2>&1 &
  FLP=$!
fi
wait $LP

free_port() { python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }

# outer_side O D: V*'s statements over the inner session $O/inner of the statement staged in $D, each proved and checked
outer_side() {
  local O=$1 D=$2 V=$S/v
  wait $FLP || { tail -30 $OUT/build-fold.log; return 1; }
  mkdir -p $O/outer; rm -rf $V; mkdir -p $V
  /usr/bin/time -v -o $O/outer/time-points.txt python3 -m verity_flock.rec_vstage points $O/inner $D/circuit.txt $D/pub-$N.bin \
    > $V/points.json 2> $O/outer/points.err || { tail -20 $O/outer/points.err; return 1; }
  /usr/bin/time -v -o $O/outer/time-fold.txt $FOLD_BIN --plain-leaves --partition $D/partition.json verity/flock-circuit \
    $D/circuit.txt $D/pub-$N.bin $V/points.json os > $V/fold.json 2> $O/outer/fold.err || { tail -20 $O/outer/fold.err; return 1; }
  cp $V/points.json $V/fold.json $O/outer/
  local ALL=$(python3 -c "
import sys; from pathlib import Path; from verity_flock import rec_outer as O, rec_vstar as V
pub, priv, proofs = V.load(Path(sys.argv[1])); print(' '.join(str(l['level']) for l in O.levels(V.queries(pub, priv, proofs))))" $O/inner)
  echo "levels: $ALL"
  vstage() {
    local name=$1; shift
    /usr/bin/time -v -o $O/outer/time-stage-$name.txt python3 -m verity_flock.rec_vstage stage $O/inner $V/fold.json $V \
      --points $V/points.json --salts $V/acc-salts.bin "$@" > $O/outer/stage-$name.json 2> $O/outer/stage-$name.err
    echo "vstage $name exit $?"
  }
  vstage verifier --levels none --no-algebra
  [ -f $V/acc-salts.bin ] || { tail -20 $O/outer/stage-verifier.err; return 1; }
  vstage alg --levels none
  for l in $ALL; do vstage L$l --levels $l --no-algebra; done
  local C
  for C in $(cd $V && find . -name inst.bin -printf '%h\n' | sed 's:^\./::' | sort -V); do
    local l=${C//\//-} P=$O/outer/${C//\//-} PIN=$(sha512sum $V/$C/circuit.txt | cut -d' ' -f1) PORT=$(free_port)
    mkdir -p $P
    $B serve --zk --listen 127.0.0.1:$PORT --out $P/sessions --instances $V/$C/pub.bin --circuit $V/$C/circuit.txt --pin $PIN \
      --operator loopback > $P/serve.log 2>&1 &
    local SP=$!
    for i in $(seq 1800); do grep -q SERVING $P/serve.log && break; kill -0 $SP 2>/dev/null || break; sleep 1; done
    /usr/bin/time -v -o $P/time-prove.txt $B prove --zk --verifier 127.0.0.1:$PORT --instances $V/$C/inst.bin \
      --circuit $V/$C/circuit.txt --warm 0 --runs 1 > $P/prove.txt 2> $P/prove.err
    echo "$l prove exit $?"
    kill $SP; wait $SP 2>/dev/null
    local SESS=$(find $P/sessions -name session.json -printf '%h\n' | sort | head -1)
    [ -n "$SESS" ] || { echo "$l: no session"; tail -5 $P/prove.err; continue; }
    $B replay --zk --instances $V/$C/pub.bin --circuit $V/$C/circuit.txt --sessions $SESS > $P/replay.txt 2>&1
    echo "$l replay exit $?"
    /usr/bin/time -v -o $P/time-lean.txt $LEAN_BIN verify --zk --statement verity/flock-circuit --circuit $V/$C/circuit.txt \
      --public $V/$C/pub.bin --partition $V/$C/partition.json --program $V/$C/program.json --registered $V/$C/registered.json \
      --session $SESS > $P/lean.log 2>&1
    echo "$l lean exit $?"
    du -sb $V/$C/inst.bin $V/$C/pub.bin $V/$C/circuit.txt > $P/sizes.txt
  done
  rm -rf $V
}

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
  $LEAN_BIN verify --statement $STMT --circuit $D/circuit.txt --public $D/pub-$N.bin --partition $D/partition.json --coins os \
    --session $SESS > $O/lean.log 2>&1
  echo "lean exit $?"
  mkdir -p $O/inner/stage && cp $D/circuit.txt $D/pub-$N.bin $O/inner/stage/
  wait $CLP
  /usr/bin/time -v -o $O/time-claims.txt $CLAIMS_BIN verity/flock-circuit $D/circuit.txt $D/pub-$N.bin $SESS os > $O/inner/claims.json 2> $O/claims.err
  echo "claims exit $?"
  [ "${OUTER:-0}" = 1 ] && { outer_side $O $D || echo "outer failed"; }
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
def verdict(p, tag):
    t = p.read_text() if p.exists() else ""
    return next((l[:300] for l in t.splitlines() if l.startswith(tag)), t[-300:])
outer = {}
for p in sorted(q for q in (o / "outer").glob("*") if q.is_dir()) if (o / "outer").exists() else []:
    lv = [json.loads(l[5:]) for l in (p / "prove.txt").read_text().splitlines() if l.startswith("LIVE")] if (p / "prove.txt").exists() else []
    outer[p.name] = {"stage": tv(f"outer/time-stage-{p.name.split('-')[0]}.txt"), "prove": tv(f"outer/{p.name}/time-prove.txt"),
                     "lean_time": tv(f"outer/{p.name}/time-lean.txt"),
                     "live": {k: lv[0].get(k) for k in ("prove_total_s", "proof_bytes", "k_log", "dense_m")} if lv else None,
                     "replay": verdict(p / "replay.txt", "UPSTREAM"), "lean": verdict(p / "lean.log", "VERDICT")}
if (o / "outer").exists():
    s["outer"] = outer | {"points": tv("outer/time-points.txt"), "fold": tv("outer/time-fold.txt")}
print(json.dumps(s))
EOF
  tail -1 $OUT/summary.jsonl
}

for c in ${CLS:?CLS="G,N_IN,N_OUT ..."}; do
  one_class "$c" || { echo "class $c failed"; exit 1; }
done
