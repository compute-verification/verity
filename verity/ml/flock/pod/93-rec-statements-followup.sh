#!/usr/bin/env bash
# Scratch job (cursor/rec-statements-integration-c4f1, never a PR): 92's follow-up on its kept files (RID=<92's run id>):
# check 5's sparse-v control (RecSparse's v one element off) staged and refused by `statement`, and `verify` of 92's
# consistently forged rec-claim RecSparse against R's opening part 0's proved session, each reason printed whole.
set -uxo pipefail
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
R0=$RESEARCH_RUN_DIR; mkdir -p $R0/out; O2=$R0/out
export RI_PINNED=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')
export RAYON_NUM_THREADS=$(echo $RI_PINNED | tr ',' '\n' | wc -l)
. verity/ml/flock/pod/rec-env.sh
P=/workspace/research/runs/$RID/out; H=$P/honest/out; ST=/workspace/jobs/gap2/$RID/honest/stage; S9=/workspace/scratch/$RID; E9=$P/e2e
IC=$ST/circuit.txt; IP=$ST/pub-16.bin; IQ=$ST/partition.json; REG=$ST/circuit-registration.json
ls -d $H/inner $ST $S9/v-c3 $S9/v-claim $E9/prove-opening-p0 || exit 1
IN=(--rec-inner-circuit $IC --rec-inner-public $IP --rec-inner-partition $IQ --rec-inner-registration $REG)
files() { echo --statement verity/flock-circuit --circuit $1/circuit.txt --public $1/pub.bin --partition $1/partition.json \
  --program $1/program.json --registered $1/registered.json; }
flags() { echo --rec-inner $E9/rec-inner-c3.json --rec-part $1 --rec-bindings $2/rec-bindings.json $( [ -f $2/rec-rows.json ] && grep -q acc $2/rec-rows.json && echo --rec-rows $2/rec-rows.json); }
python3 - $S9/v-c3/sparse <<'PY'
import json, sys
from pathlib import Path
from verity_flock import rec_bindings as RB
c = Path(sys.argv[1])
(c / "rec-bindings.json").write_text(json.dumps(RB.bindings(json.loads((c / "partition.json").read_text()), json.loads((c / "program.json").read_text()))))
PY
python3 verity/ml/flock/pod/c3_e2e.py stage sparse-v --inner $H/inner --fold $H/fold.json --points $H/points.json --circuit $IC \
  --salts $S9/acc-salts-c3.bin --out $S9/v-sparse-v > $O2/stage-sparse-v.json 2> $O2/stage-sparse-v.err
echo "stage sparse-v exit $?" | tee -a $O2/exits.txt
D=$S9/v-sparse-v/sparse; cp $S9/v-c3/sparse/rec-bindings.json $D/
$LEAN_BIN statement $(files $D) $(flags 6 $D) "${IN[@]}" --rec-inner-session $H/inner/prover --rec-inner-transcript $H/inner/public.json \
  > $O2/sparse-v.out 2> $O2/sparse-v.err
echo "statement sparse-v exit $? (1: refused)" | tee -a $O2/exits.txt; head -c 2000 $O2/sparse-v.out
D=$S9/v-claim/sparse
SESS=$(find $E9/prove-opening-p0/sessions -name session.json -printf '%h\n' | sort | head -1)
$LEAN_BIN verify --zk $(files $D) $(flags 6 $D) "${IN[@]}" --rec-inner-session $H/inner/prover \
  --rec-inner-transcript $H/inner/public.json --session $SESS > $O2/claim-verify-sparse.log 2>&1
echo "verify claim rec-sparse exit $? (1: refused)" | tee -a $O2/exits.txt
grep -o '"why":"[^"]*"' $O2/claim-verify-sparse.log | tee $O2/claim-verify-sparse.why
cat $O2/exits.txt
