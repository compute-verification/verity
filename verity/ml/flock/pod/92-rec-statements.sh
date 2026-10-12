#!/usr/bin/env bash
# Scratch job (cursor/rec-statements-integration-c4f1, never a PR): main + cursor/rec-vstar-statements-8c79 (candidate 3's V*
# list: RecSparse, R's levels as rec-open-stacked/l<L>-h<H>, R's opening at META's stacked) over 91's integration
# (cursor/rec-r-shape-integration-8e55). D12 again, on candidate 3's whole V* list:
# 88-rec-c3-inner.sh N=16 SPARSE=18,13,6,0 STACKED=1; rec-inner --sparse 18,13,6,0 (and its refusal without --sparse);
# rec-statements (the 11); rec-transcript --rec-inner on both records (checkCovers); L3 and RecSparse + R staged
# (c3_e2e.py stage level3 / session); Lean `statement` on each under candidate 3's inner shape with the inner transcript;
# L3 and R's L2 proved by C-Flock --zk under serve --draw subset:N, replayed by upstream, and Lean's verify under
# --rec-inner (plain, --sparse) with --rec-inner-transcript and --rec-inner-registration on each record.
#   research run --on vy-cpu-1 --mem-gb 40 --source . --cwd source --declared-output 'out/**' -- \
#     python3 tools/verity/check/slot.py --by gap2 -- bash $0
set -uxo pipefail
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
R0=$RESEARCH_RUN_DIR; RID=${RESEARCH_RUN_ID:-local-$$}; mkdir -p $R0/out
K=/workspace/jobs/gap2/$RID; mkdir -p $K || exit 1
RESEARCH_RUN_DIR=$R0/out/honest RESEARCH_RUN_ID=$RID-honest KEEP=$K/honest bash verity/ml/flock/pod/88-rec-c3-inner.sh N=16 \
  SPARSE=18,13,6,0 STACKED=1 > $R0/out/88.log 2>&1
echo "88 exit $?" | tee -a $R0/out/exits.txt

export RI_PINNED=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')
export RAYON_NUM_THREADS=$(echo $RI_PINNED | tr ',' '\n' | wc -l)
. verity/ml/flock/pod/rec-env.sh
wait $LP
E=$O/e2e; mkdir -p $E; X=$O/exits.txt
H=$O/honest/out; ST=$K/honest/stage; IC=$ST/circuit.txt; IP=$ST/pub-16.bin; IQ=$ST/partition.json; IG=$ST/program.json
REG=$ST/circuit-registration.json
[ -f $H/inner/session/session.json ] && [ -f $IQ ] || { echo "no inner session or partition" | tee -a $X; exit 1; }
sha256sum $LEAN_BIN $B > $E/bins.sha256
head -2 $IC > $E/circuit-meta.txt; cp $REG $E/

# the verifier's inner shape: refused without the bucket's RecSparse shape, since META names R
$LEAN_BIN rec-inner --circuit $IC --public $IP --partition $IQ --program $IG > $E/rec-inner-nosparse.json 2> $E/rec-inner-nosparse.err
echo "rec-inner without --sparse exit $? (1: refused)" | tee -a $X
$LEAN_BIN rec-inner --circuit $IC --public $IP --partition $IQ --program $IG --sparse 18,13,6,0 > $E/rec-inner-c3.json 2> $E/rec-inner.err
echo "rec-inner --sparse exit $?" | tee -a $X
RI=$E/rec-inner-c3.json
$LEAN_BIN rec-statements --rec-inner $RI > $E/rec-statements.txt 2>&1
echo "rec-statements exit $?" | tee -a $X

for s in session prover; do
  $LEAN_BIN rec-transcript --rec-inner-transcript $H/inner/public.json --rec-inner-session $H/inner/$s --rec-inner-circuit $IC \
    --rec-inner-registration $REG --rec-inner $RI > $E/rec-transcript-$s.txt 2>&1
  echo "rec-transcript --rec-inner $s exit $?" | tee -a $X
done

SALTS=$S/acc-salts.bin
python3 verity/ml/flock/pod/c3_e2e.py stage level3 --inner $H/inner --fold $H/fold.json --points $H/points.json --circuit $IC \
  --salts $SALTS --out $S/v-l3 > $E/stage-l3.json 2> $E/stage-l3.err
echo "stage l3 exit $?" | tee -a $X
python3 verity/ml/flock/pod/c3_e2e.py stage session --inner $H/inner --fold $H/fold.json --points $H/points.json --circuit $IC \
  --salts $SALTS --out $S/v-c3 > $E/stage-c3.json 2> $E/stage-c3.err
echo "stage sparse+R exit $?" | tee -a $X

# each staged statement's verifier-own flags: its index in the list, its bindings, its rec-acc rows
own() {  # own DIR -> the --rec-* flags of DIR's statement
  python3 - $E $1 <<'EOF'
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
print(f"--rec-inner {e}/rec-inner-c3.json --rec-part {names.index(unit)} --rec-bindings {c}/rec-bindings.json --rec-rows {c}/rec-rows.json")
EOF
}
IN=(--rec-inner-circuit $IC --rec-inner-public $IP --rec-inner-partition $IQ --rec-inner-registration $REG)
files() { echo --statement verity/flock-circuit --circuit $1/circuit.txt --public $1/pub.bin --partition $1/partition.json \
  --program $1/program.json --registered $1/registered.json; }

: > $E/statements.tsv
for d in $S/v-l3/L3 $(find $S/v-c3 -name circuit.txt -printf '%h\n' | sort); do
  u=$(head -1 $d/circuit.txt | cut -d' ' -f2)
  flags=$(own $d) || { echo "own $d failed" | tee -a $X; continue; }
  for s in prover session; do
    $LEAN_BIN statement $(files $d) $flags "${IN[@]}" --rec-inner-session $H/inner/$s --rec-inner-transcript $H/inner/public.json \
      > $E/st.out 2> $E/st.err
    rc=$?
    printf '%s\t%s\t%s\t%s\t%s\n' "$u" "$s" "$rc" "$(head -1 $E/st.out | cut -c1-400)" "$(tail -1 $E/st.err | cut -c1-300)" >> $E/statements.tsv
    echo "statement $u $s exit $rc" | tee -a $X
  done
done

prove() {  # prove TAG DIR: serve --zk under the draw, prove, replay, and Lean's verify on each record
  local t=$1 C=$2 D=$E/prove-$1; mkdir -p $D
  local NI=$(python3 -c 'import json, sys; print(json.loads(open(sys.argv[1], "rb").readline())["instances"])' $C/pub.bin)
  local DRAW=${DRAW:-subset:$NI}; echo "$DRAW" > $D/draw.txt
  local PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
  local PIN=$(sha512sum $C/circuit.txt | cut -d' ' -f1)
  $B serve --zk --listen 127.0.0.1:$PORT --out $D/sessions --instances $C/pub.bin --circuit $C/circuit.txt --pin $PIN --draw $DRAW \
    --operator loopback > $D/serve.log 2>&1 &
  local SP=$!
  for i in $(seq 1800); do grep -q SERVING $D/serve.log && break; kill -0 $SP 2>/dev/null || break; sleep 1; done
  /usr/bin/time -v -o $D/time-prove.txt $B prove --zk --verifier 127.0.0.1:$PORT --instances $C/inst.bin --circuit $C/circuit.txt \
    --warm 0 --runs 1 > $D/prove.txt 2> $D/prove.err
  echo "prove $t exit $?" | tee -a $X
  kill $SP; wait $SP 2>/dev/null
  local SESS=$(find $D/sessions -name session.json -printf '%h\n' | sort | head -1)
  [ -n "$SESS" ] || { tail -20 $D/serve.log $D/prove.err; return 1; }
  $B replay --zk --instances $C/pub.bin --circuit $C/circuit.txt --sessions $SESS > $D/replay.txt 2>&1; echo "replay $t exit $?" | tee -a $X
  local flags=$(own $C)
  for s in prover session; do
    /usr/bin/time -v -o $D/time-lean-$s.txt $LEAN_BIN verify --zk $(files $C) $flags "${IN[@]}" --rec-inner-session $H/inner/$s \
      --rec-inner-transcript $H/inner/public.json --session $SESS > $D/lean-$s.log 2>&1
    echo "lean $t $s exit $?" | tee -a $X
  done
  grep -h "^VERDICT" $D/lean-*.log | cut -c1-800 >> $E/verdicts.txt
  for f in $D/lean-*.log; do echo "== $f"; grep -v "^VERDICT" $f | tail -3; done >> $E/verdicts.txt
}
prove l3 $S/v-l3/L3
for d in $(find $S/v-c3 -name circuit.txt -printf '%h\n'); do
  [ "$(head -1 $d/circuit.txt | cut -d' ' -f2)" = rec-open-stacked/l16-h1 ] && RL2=$d
done
prove r-l2 ${RL2:?no staged rec-open-stacked/l16-h1}
cat $X
