#!/usr/bin/env bash
# Scratch job (cursor/rec-statements-integration-c4f1, never a PR): main + cursor/rec-vstar-statements-8c79 (candidate 3's V*
# list: RecSparse, R's levels as rec-open-stacked/l<L>-h<H>, R's opening at META's stacked) over 91's integration
# (cursor/rec-r-shape-integration-8e55). D12 again, on candidate 3's whole V* list:
# 88-rec-c3-inner.sh N=16 SPARSE=18,13,6,0 STACKED=1; rec-inner --sparse 18,13,6,0 (and its refusal without --sparse);
# rec-statements (the 11); rec-transcript --rec-inner on both records (checkCovers); L3 and RecSparse + R staged
# (c3_e2e.py stage level3 / session); Lean `statement` on each under candidate 3's inner shape with the inner transcript;
# L3 and R's L2 proved by C-Flock --zk under serve --draw subset:N, replayed by upstream, and Lean's verify under
# --rec-inner (plain, --sparse) with --rec-inner-transcript and --rec-inner-registration on each record. Gap 4 and check 5:
# c3_e2e.py's forgeries (claim, claim-own, sparse-v, stacked-coef, opening-v), each refused by `statement` by name, the
# claim forgery by `verify` too; R's opening part 0 proved; the fixture rec-c3-inner (out/fixture) and rec-coins' record.
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

python3 verity/ml/flock/pod/c3_e2e.py stage level3 --inner $H/inner --fold $H/fold.json --points $H/points.json --circuit $IC \
  --salts $S/acc-salts-l3.bin --out $S/v-l3 > $E/stage-l3.json 2> $E/stage-l3.err
echo "stage l3 exit $?" | tee -a $X
python3 verity/ml/flock/pod/c3_e2e.py stage session --inner $H/inner --fold $H/fold.json --points $H/points.json --circuit $IC \
  --salts $S/acc-salts-c3.bin --out $S/v-c3 > $E/stage-c3.json 2> $E/stage-c3.err
echo "stage sparse+R exit $?" | tee -a $X

# (the relay's record, session/, is refused by name: check 5 reads the prover's)
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
# RecSparse reads no rec-acc and shares no row: it takes no --rec-rows
print(f"--rec-inner {e}/rec-inner-c3.json --rec-part {names.index(unit)} --rec-bindings {c}/rec-bindings.json"
      + (f" --rec-rows {c}/rec-rows.json" if rows else ""))
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
# the inner session as the fixture rec-c3-inner holds it, with check 5's record of the inner statement (rec-coins)
FX=$O/fixture/rec-c3-inner; mkdir -p $FX/prover $FX/session
cp $H/inner/public.json $H/inner/private.json $H/fold.json $H/points.json $E/circuit-meta.txt $REG $FX/
cp $H/inner/prover/session.json $FX/prover/; cp $H/inner/session/session.json $FX/session/
$LEAN_BIN rec-coins --rec-inner-circuit $IC --rec-inner-public $IP --rec-inner-partition $IQ --rec-inner-session $H/inner/prover \
  > $FX/rec-coins.json 2> $E/rec-coins.err
echo "rec-coins exit $?" | tee -a $X

# the controls: each forgery staged, and `statement` on each statement it changed (prover record, inner transcript)
: > $E/controls.tsv
for k in claim claim-own sparse-v stacked-coef opening-v; do
  python3 verity/ml/flock/pod/c3_e2e.py stage $k --inner $H/inner --fold $H/fold.json --points $H/points.json --circuit $IC \
    --salts $S/acc-salts-c3.bin --out $S/v-$k > $E/stage-$k.json 2> $E/stage-$k.err
  echo "stage $k exit $?" | tee -a $X
  for d in $(find $S/v-$k -name circuit.txt -printf '%h\n' | sort); do
    rel=${d#$S/v-$k/}
    cmp -s $d/registered.json $S/v-c3/$rel/registered.json && cmp -s $d/pub.bin $S/v-c3/$rel/pub.bin && continue
    u=$(head -1 $d/circuit.txt | cut -d' ' -f2)
    flags=$(own $d) || { echo "own $d failed" | tee -a $X; continue; }
    $LEAN_BIN statement $(files $d) $flags "${IN[@]}" --rec-inner-session $H/inner/prover --rec-inner-transcript $H/inner/public.json \
      > $E/st.out 2> $E/st.err
    rc=$?
    printf '%s\t%s\t%s\t%s\t%s\n' "$k" "$u" "$rc" "$(head -1 $E/st.out | cut -c1-500)" "$(tail -1 $E/st.err | cut -c1-300)" >> $E/controls.tsv
    echo "control $k $u exit $rc (1: refused)" | tee -a $X
    echo "$k $d" >> $E/control-dirs.txt
  done
done

prove l3 $S/v-l3/L3
for d in $(find $S/v-c3 -name circuit.txt -printf '%h\n'); do
  case "$(head -1 $d/circuit.txt | cut -d' ' -f2)" in
    rec-open-stacked/l16-h1) RL2=$d ;;
    rec-opening/*-p0) OP0=$d ;;
  esac
done
prove r-l2 ${RL2:?no staged rec-open-stacked/l16-h1}
[ -n "${OP0:-}" ] && prove opening-p0 $OP0

# gap 4 at verify: the consistently registered wrong claim, set up against a proved session (R's opening part 0's), is
# refused at setup in RecSparse and R's opening alike
SESS=$(find $E/prove-opening-p0/sessions -name session.json -printf '%h\n' 2>/dev/null | sort | head -1)
: > $E/claim-verify.txt
for d in $(grep '^claim ' $E/control-dirs.txt | cut -d' ' -f2); do
  [ -n "$SESS" ] || break
  u=$(head -1 $d/circuit.txt | cut -d' ' -f2)
  $LEAN_BIN verify --zk $(files $d) $(own $d) "${IN[@]}" --rec-inner-session $H/inner/prover \
    --rec-inner-transcript $H/inner/public.json --session $SESS > $E/cv.log 2>&1
  rc=$?
  echo "$u exit $rc $(grep -h '^VERDICT' $E/cv.log | cut -c1-600)" >> $E/claim-verify.txt
  echo "verify claim $u exit $rc (1: refused)" | tee -a $X
done
cat $X
