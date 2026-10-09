#!/usr/bin/env bash
# rec-private (lane rec-v0): the recursion's overhead's baseline. The private circuit 87-rec-private.sh registers for each
# class (the same netlist and x: CLS, SEED, REC_NETLIST), as a public circuit proved directly, with no recursion:
# rec_inner.py direct stages it with C-Flock's salted leaves, and rec-private-outer.sh's outer proves it with --zk under the
# verifier's draw of every instance, Lean's verify --zk deciding the session and the live build replaying it. out/baseline.json:
# each class's staging, live record, Lean verdict and times, read as rec_private_summary.py reads an outer session.
# Run from the verity tree (research run --cwd source).
# env: RUNS=16,16,8:64;1024,1024,512:8 (CLS:N, ;-separated)  SEED=0  CPUS=32 (as 87's)  FLOCK_WORK (as 87)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
if [ -z "${RI_PINNED:-}" ]; then
  . verity/ml/flock/pod/cpu-slices.sh
  hold_slices "" "${CPUS:-32}" || exit 1
  export RI_PINNED=$SLICE_CPUS RAYON_NUM_THREADS=$(echo $SLICE_CPUS | tr ',' '\n' | wc -l)
  exec taskset -c $SLICE_CPUS bash "$0" "$@"
fi
export FC_COINS=os FLOCK_WORK=${FLOCK_WORK:-/workspace/jobs/rec-v0}
. verity/ml/flock/pod/rec-env.sh
SEED=${SEED:-0}; RUNS=${RUNS:-16,16,8:64;1024,1024,512:8}
PY=verity/ml/flock/pod/rec_inner.py
wait $LP; echo "flock-verify build exit $?"; [ -x $LEAN_BIN ] || { tail -20 $O/build-lean.log; exit 1; }
. verity/ml/flock/pod/rec-private-outer.sh
NAMES=()
for r in ${RUNS//;/ }; do
  CLS=${r%:*}; N=${r#*:}; NAME=direct-${CLS//,/-}-n$N; DD=$S/$NAME
  /usr/bin/time -v -o $O/time-stage-$NAME.txt python3 $PY direct --cls $CLS --n $N --seed $SEED --out $DD > $O/stage-$NAME.json \
    2> $O/stage-$NAME.err || { tail -20 $O/stage-$NAME.err; continue; }
  outer $NAME $DD plain
  NAMES+=($NAME)
done
python3 - $O "${NAMES[@]}" <<'EOF'
import json, sys
from pathlib import Path
sys.path.insert(0, "verity/ml/flock/pod")
import rec_private_summary as R

o = Path(sys.argv[1])
b = {}
for n in sys.argv[2:]:
    d = o / "outer" / n
    b[n] = {"stage": json.loads(R.text(o / f"stage-{n}.json") or "null"), "stage_wall": R.tv(o / f"time-stage-{n}.txt", r"Elapsed \(wall clock\) time"),
            "live": R.live(d), "lean": R.lean(d / "lean.log"), "lean_wall": R.tv(d / "time-lean.txt", r"Elapsed \(wall clock\) time"),
            "replay": R.up(d / "replay.txt"), "sizes": R.text(d / "sizes.txt")}
    print("BASELINE", n, json.dumps({"live": b[n]["live"]["verdict"], "lean": b[n]["lean"]["verdict"], "prove_s": b[n]["live"]["prove_s"],
                                     "proof_bytes": b[n]["live"]["proof_bytes"], "lean_wall": b[n]["lean_wall"]}))
(o / "baseline.json").write_text(json.dumps(b, indent=1))
EOF
