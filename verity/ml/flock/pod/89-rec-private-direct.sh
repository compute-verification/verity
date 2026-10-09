#!/usr/bin/env bash
# rec-private (lane rec-v0): V*'s honest statements that 87-rec-private.sh staged, each proved outside the recursive session
# (rec-private-outer.sh's plain mode: the firewall released none past its first part, whose verdict is its form's refusal),
# for each verifier's verdict on it; the release of an earlier run's parts decided again by Lean's core-firewall (RECHECK);
# then the public views of the two netlists (88-rec-private-public.sh) and out/summary.json (rec_private_summary.py).
# Run after 87 on its $FLOCK_WORK/data/rec-private/$TAG (research run --cwd source).
# env: TAG=16-16-8-n64  DIRECT (parts, default every part of the cap's)  RECHECK (an earlier run's out/outer: each part
#      behind the firewall there, its record and transcript)  CPUS=32  FLOCK_WORK (as 87)
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
TAG=${TAG:-16-16-8-n64}; DD=$FW/data/rec-private/$TAG; V=$DD/v
[ -f $DD/cap.json ] && [ -d $V ] || { echo "no 87 staging under $DD"; exit 1; }
cp $DD/cap.json $O/
TSHA=$( (find verity/core/service/flock verity/core/service/firewall verity/core/protocols/interface verity/core/difftest/lean \
  -name .lake -prune -o -type f -print | LC_ALL=C sort | xargs sha256sum; cat verity/ml/flock/tests/lean/InnerFold.lean) | sha256sum | cut -c1-16)
T=$L/tools-$TSHA
[ -x $T/core-firewall ] || { echo "no Lean tools at $T (87 builds them)"; exit 1; }
wait $LP; [ -x $LEAN_BIN ] || { tail -20 $O/build-lean.log; exit 1; }
. verity/ml/flock/pod/rec-private-outer.sh
SID=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['session'])" $DD/inner-honest/firewall.json) || exit 1

PARTS=${DIRECT:-$(python3 -c "import json, sys; print(','.join(json.load(open(sys.argv[1]))['parts']))" $DD/cap.json)}
for p in ${PARTS//,/ }; do
  case $p in L*) C=$V/$p ;; alg-p*) C=$V/alg/p${p#alg-p} ;; esac
  outer $p-direct $C plain
done
if [ -n "${RECHECK:-}" ]; then
  for R in $RECHECK/*/; do
    [ -f $R/record-before.jsonl ] && [ -d $R/transcript ] || continue
    D=$O/outer/$(basename $R)-recheck; mkdir -p $D && cp -r $R/record-before.jsonl $R/transcript $D/
    release_check $D $SID $(basename $R); echo "$(basename $R) core-firewall recheck exit $?"
  done
fi
bash verity/ml/flock/pod/88-rec-private-public.sh TAG=$TAG > $O/public.log 2>&1; echo "public views exit $?"
python3 verity/ml/flock/pod/rec_private_summary.py $O
true
