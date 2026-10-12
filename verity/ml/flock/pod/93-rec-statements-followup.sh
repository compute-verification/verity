#!/usr/bin/env bash
# Scratch job (cursor/rec-statements-integration-c4f1, never a PR): 92's follow-up on its kept honest run (RID=<92's run id>):
# check 5's sparse-v control (RecSparse's v one element off) staged and refused by `statement`, and `verify` of the
# consistently forged rec-claim RecSparse against R's opening part 0's proved session, each reason printed whole. 92's
# scratch (its stagings and salts) went with its exit, so the honest session, sparse-v and claim are staged again here.
set -uxo pipefail
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
R0=$RESEARCH_RUN_DIR; mkdir -p $R0/out; O2=$R0/out
export RI_PINNED=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')
export RAYON_NUM_THREADS=$(echo $RI_PINNED | tr ',' '\n' | wc -l)
. verity/ml/flock/pod/rec-env.sh
wait $LP
P=/workspace/research/runs/$RID/out; H=$P/honest/out; ST=/workspace/jobs/gap2/$RID/honest/stage; E9=$P/e2e
IC=$ST/circuit.txt; IP=$ST/pub-16.bin; IQ=$ST/partition.json; REG=$ST/circuit-registration.json
X=$O2/exits.txt; : > $X
ls -d $H/inner $ST $E9/rec-inner-c3.json $E9/rec-statements.txt $E9/prove-opening-p0 || exit 1
IN=(--rec-inner-circuit $IC --rec-inner-public $IP --rec-inner-partition $IQ --rec-inner-registration $REG)
files() { echo --statement verity/flock-circuit --circuit $1/circuit.txt --public $1/pub.bin --partition $1/partition.json \
  --program $1/program.json --registered $1/registered.json; }
own() {  # own DIR -> the --rec-* flags of DIR's statement (92's)
  python3 - $E9 $1 <<'EOF'
import json, sys
from pathlib import Path
from verity_flock import rec_bindings as RB
e, c = map(Path, sys.argv[1:])
(line,) = [json.loads(ln[9:]) for ln in (e / "rec-statements.txt").read_text().splitlines() if ln.startswith("RECVSTAR ")]
names = [s[0] for s in line["statements"]]
with (c / "circuit.txt").open() as f:
    unit = f.readline().split()[1]
ports = json.loads((c / "registered.json").read_text())["ports"]
rows = {"acc0": ports["acc"]} if "acc" in ports else {nm: x for nm, x in ports.items() if nm in ("acc0", "acc1")}
(c / "rec-rows.json").write_text(json.dumps({"format": "verity/registered-reads/v0", "ports": rows}))
(c / "rec-bindings.json").write_text(json.dumps(RB.bindings(json.loads((c / "partition.json").read_text()),
                                                             json.loads((c / "program.json").read_text()))))
print(f"--rec-inner {e}/rec-inner-c3.json --rec-part {names.index(unit)} --rec-bindings {c}/rec-bindings.json"
      + (f" --rec-rows {c}/rec-rows.json" if rows else ""))
EOF
}
for k in session sparse-v claim; do
  python3 verity/ml/flock/pod/c3_e2e.py stage $k --inner $H/inner --fold $H/fold.json --points $H/points.json --circuit $IC \
    --salts $S/acc-salts-c3.bin --out $S/v-$k > $O2/stage-$k.json 2> $O2/stage-$k.err
  echo "stage $k exit $?" | tee -a $X
done
D=$(find $S/v-sparse-v -name circuit.txt -printf '%h\n' | while read d; do head -1 $d/circuit.txt | grep -q ' rec-sparse/' && echo $d; done | head -1)
$LEAN_BIN statement $(files $D) $(own $D) "${IN[@]}" --rec-inner-session $H/inner/prover --rec-inner-transcript $H/inner/public.json \
  > $O2/sparse-v.out 2> $O2/sparse-v.err
echo "statement sparse-v exit $? (1: refused)" | tee -a $X; head -c 2000 $O2/sparse-v.out; tail -3 $O2/sparse-v.err
SESS=$(find $E9/prove-opening-p0/sessions -name session.json -printf '%h\n' | sort | head -1)
for d in $(find $S/v-claim -name circuit.txt -printf '%h\n' | sort); do
  rel=${d#$S/v-claim/}
  cmp -s $d/registered.json $S/v-session/$rel/registered.json && cmp -s $d/pub.bin $S/v-session/$rel/pub.bin && continue
  u=$(head -1 $d/circuit.txt | cut -d' ' -f2)
  $LEAN_BIN verify --zk $(files $d) $(own $d) "${IN[@]}" --rec-inner-session $H/inner/prover \
    --rec-inner-transcript $H/inner/public.json --session $SESS > $O2/cv-${u//\//_}.out 2> $O2/cv-${u//\//_}.err
  rc=$?
  echo "$u exit $rc $(grep -ho '"why":"[^"]*"' $O2/cv-${u//\//_}.out)" >> $O2/claim-verify.txt
  echo "verify claim $u exit $rc (1: refused)" | tee -a $X
done
# gap 4 at verify on RecSparse's own proof: the forged RecSparse proved under its own draw, so verify reaches its rec-pub
CS=$(find $S/v-claim -name circuit.txt -printf '%h\n' | while read d; do head -1 $d/circuit.txt | grep -q ' rec-sparse/' && echo $d; done | head -1)
D=$O2/prove-claim-sparse; mkdir -p $D
NI=$(python3 -c 'import json, sys; print(json.loads(open(sys.argv[1], "rb").readline())["instances"])' $CS/pub.bin)
PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
PIN=$(sha512sum $CS/circuit.txt | cut -d' ' -f1)
$B serve --zk --listen 127.0.0.1:$PORT --out $D/sessions --instances $CS/pub.bin --circuit $CS/circuit.txt --pin $PIN --draw subset:$NI \
  --operator loopback > $D/serve.log 2>&1 &
SP=$!
for i in $(seq 1800); do grep -q SERVING $D/serve.log && break; kill -0 $SP 2>/dev/null || break; sleep 1; done
$B prove --zk --verifier 127.0.0.1:$PORT --instances $CS/inst.bin --circuit $CS/circuit.txt --warm 0 --runs 1 > $D/prove.txt 2> $D/prove.err
echo "prove forged claim rec-sparse exit $?" | tee -a $X
kill $SP; wait $SP 2>/dev/null
CSESS=$(find $D/sessions -name session.json -printf '%h\n' | sort | head -1)
if [ -n "$CSESS" ]; then
  $B replay --zk --instances $CS/pub.bin --circuit $CS/circuit.txt --sessions $CSESS > $D/replay.txt 2>&1; echo "replay forged claim rec-sparse exit $?" | tee -a $X
  $LEAN_BIN verify --zk $(files $CS) $(own $CS) "${IN[@]}" --rec-inner-session $H/inner/prover \
    --rec-inner-transcript $H/inner/public.json --session $CSESS > $D/lean.out 2> $D/lean.err
  echo "verify forged claim rec-sparse on its own proof exit $? (1: refused)" | tee -a $X
  grep -ho '"why":"[^"]*"' $D/lean.out | tee -a $O2/claim-verify.txt
else
  tail -20 $D/serve.log $D/prove.err
fi
cat $X $O2/claim-verify.txt
