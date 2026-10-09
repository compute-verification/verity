# rec-private (lane rec-v0): Lean's verify --zk of this tree, again, on outer sessions earlier runs on this node recorded
# (87-rec-private.sh's, 91-rec-private-baseline.sh's): for each part of each run, the statement its sizes.txt names
# (circuit.txt, pub.bin and the verifier's partition, program and registered reads beside them) and its recorded session,
# under the verifier this tree builds. out/reverify/RUN/PART/lean.{log,err}, and out/reverify.json with each part's verdict
# now beside the run's own.
# env: RUNS (run ids, comma-separated)  PARTS (default every part)  JOBS=8 (verifies at once)  CPUS=16
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
if [ -z "${RI_PINNED:-}" ]; then
  . verity/ml/flock/pod/cpu-slices.sh
  hold_slices "" "${CPUS:-16}" || exit 1
  export RI_PINNED=$SLICE_CPUS RAYON_NUM_THREADS=$(echo $SLICE_CPUS | tr ',' '\n' | wc -l)
  exec taskset -c $SLICE_CPUS bash "$0" "$@"
fi
export FLOCK_WORK=${FLOCK_WORK:-/workspace/jobs/rec-v0}
. verity/ml/flock/pod/rec-env.sh
wait $LP || { tail -20 $O/build-lean.log; exit 1; }
[ -x $LEAN_BIN ] || { tail -20 $O/build-lean.log; exit 1; }
ROOT=${RESEARCH_RUNS_ROOT:-/workspace/research/runs}

one() {
  local R=$1 NAME=$2 D=$ROOT/$1/out/outer/$2 E=$O/reverify/$1/$2 C SESS VF=()
  mkdir -p $E
  C=$(dirname "$(grep -m1 '/circuit.txt$' $D/sizes.txt | cut -f2)")
  SESS=$(find $D/sessions -name session.json -printf '%h\n' 2>/dev/null | sort | head -1)
  { [ -s $C/circuit.txt ] && [ -n "$SESS" ]; } || { echo "$R $NAME: no statement or session"; return 0; }
  for f in program registered; do [ -f $C/$f.json ] && VF+=(--$f $C/$f.json); done
  /usr/bin/time -v -o $E/time-lean.txt $LEAN_BIN verify --zk --statement verity/flock-circuit --circuit $C/circuit.txt --public $C/pub.bin \
    --partition $C/partition.json "${VF[@]}" --session $SESS > $E/lean.log 2> $E/lean.err
  echo "$R $NAME lean exit $?"
}
export -f one
export O LEAN_BIN ROOT
for R in ${RUNS//,/ }; do
  for D in $ROOT/$R/out/outer/*/; do
    N=$(basename $D)
    [ -z "${PARTS:-}" ] || [[ ",$PARTS," == *",$N,"* ]] || continue
    echo "$R $N"
  done
done | xargs -P ${JOBS:-8} -L 1 bash -c 'one $0 $1'

PYTHONPATH=verity/ml/flock/pod:$PYTHONPATH python3 - $O $ROOT ${RUNS//,/ } <<'EOF'
import json, sys
from pathlib import Path

import rec_private_summary as S

out, root, runs = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
res = {}
for r in runs:
    for d in sorted((out / "reverify" / r).glob("*/")):
        res.setdefault(r, {})[d.name] = {"now": S.lean(d / "lean.log")["verdict"],
                                         "recorded": S.lean(root / r / "out" / "outer" / d.name / "lean.log")["verdict"],
                                         "lean_wall": S.tv(d / "time-lean.txt", r"Elapsed \(wall clock\) time")}
(out / "reverify.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
for r, parts in res.items():
    for n, v in parts.items():
        print("REVERIFY", r, n, json.dumps(v, ensure_ascii=False))
EOF
