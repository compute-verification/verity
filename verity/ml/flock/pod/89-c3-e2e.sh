#!/usr/bin/env bash
# Scratch job (cursor/c3-e2e-integration-8c79, never a PR): candidate 3's outer V* session end to end at Shape(25,22,8).
# 88-rec-c3-inner.sh N=16 SPARSE=18,13,6,0 STACKED=1 honest, FORGE=cell and FORGE=point (b907's run), then c3_e2e.py: every
# outer statement staged (rec_vstage) and checked by Lean's flock-verify statement, the forgeries (rec-claim, h, R's cell and
# point), rec-transcript on the five-stream transcript, and the smallest statement proved (serve/prove --zk) and verified by
# Lean. Outputs under out/: honest/ cell/ point/ (88's out/), e2e/ (stage-*.json, lean.jsonl, prove-*/), summary lines.
#   research run --on vy-cpu-1 --source . --cwd source -- python3 tools/verity/check/slot.py --by c3-e2e -- bash $0
set -uxo pipefail
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
R0=$RESEARCH_RUN_DIR; RID=${RESEARCH_RUN_ID:-local-$$}; mkdir -p $R0/out
# PREV=RUN_ID: an earlier run of this job on this machine, whose 88 sessions (and kept inner statements) are reused
if [ -n "${PREV:-}" ]; then
  K=/workspace/jobs/c3e2e/$PREV
  for f in honest cell point; do cp -r /workspace/research/runs/$PREV/out/$f $R0/out/ || exit 1; done
  echo "reused $PREV" | tee -a $R0/out/exits.txt
else
  K=/workspace/jobs/c3e2e/$RID; mkdir -p $K || exit 1
  for f in honest cell point; do
    case $f in honest) X=;; *) X="FORGE=$f";; esac
    RESEARCH_RUN_DIR=$R0/out/$f RESEARCH_RUN_ID=$RID-$f KEEP=$K/$f bash verity/ml/flock/pod/88-rec-c3-inner.sh N=16 SPARSE=18,13,6,0 STACKED=1 $X \
      > $R0/out/88-$f.log 2>&1
    echo "88 $f exit $?" | tee -a $R0/out/exits.txt
  done
fi

export RI_PINNED=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')
export RAYON_NUM_THREADS=$(echo $RI_PINNED | tr ',' '\n' | wc -l)
. verity/ml/flock/pod/rec-env.sh
wait $LP
STAGES=${STAGES:-honest claim claim-own h cell point}
E=$O/e2e; mkdir -p $E; LJ=$E/lean.jsonl; SALTS=$S/acc-salts.bin
H=$O/honest/out; ST=$K/honest/stage; IC=$ST/circuit.txt; IP=$ST/pub-16.bin
P="python3 verity/ml/flock/pod/c3_e2e.py"
sha256sum $LEAN_BIN > $E/lean-bin.sha256

$LEAN_BIN rec-transcript --rec-inner-transcript $H/inner/public.json --rec-inner-session $H/inner/session > $E/rec-transcript.txt 2>&1
echo "rec-transcript exit $?" | tee -a $O/exits.txt

st() {  # st KIND INNER_OUT DIR
  /usr/bin/time -v -o $E/time-stage-$1.txt $P stage $1 --inner $2/inner --fold $2/fold.json --points $2/points.json --circuit $IC \
    --salts $SALTS --out $3 > $E/stage-$1.json 2> $E/stage-$1.err
  echo "stage $1 exit $?" | tee -a $O/exits.txt
}
if [[ " $STAGES " == *" honest "* ]]; then
  st honest $H $S/v-honest
  $P lean --bin $LEAN_BIN --dir $S/v-honest --inner-circuit $IC --inner-public $IP --out $LJ --label honest > $E/lean-honest.log 2>&1
  echo "lean honest exit $?" | tee -a $O/exits.txt
fi
for k in claim claim-own h; do
  [[ " $STAGES " == *" $k "* ]] || continue
  SALTS=$S/acc-salts-$k.bin st $k $H $S/v-$k
  $P lean --bin $LEAN_BIN --dir $S/v-$k --inner-circuit $IC --inner-public $IP --out $LJ --label $k > $E/lean-$k.log 2>&1
  echo "lean $k exit $?" | tee -a $O/exits.txt
  rm -rf $S/v-$k
done
for f in cell point; do
  [[ " $STAGES " == *" $f "* ]] || continue
  [ -f $O/$f/out/fold.json ] || [ -f $O/$f/out/points.json ] || { echo "no $f session"; continue; }
  FO=$O/$f/out; [ -f $FO/fold.json ] || FO=$H   # FORGE runs share the honest fold only if theirs is missing
  /usr/bin/time -v -o $E/time-stage-$f.txt $P stage session --inner $O/$f/out/inner --fold $FO/fold.json --points $FO/points.json \
    --circuit $IC --salts $S/acc-salts-$f.bin --out $S/v-$f > $E/stage-$f.json 2> $E/stage-$f.err
  echo "stage $f exit $?" | tee -a $O/exits.txt
  $P lean --bin $LEAN_BIN --dir $S/v-$f --inner-circuit $IC --inner-public $IP --out $LJ --label $f > $E/lean-$f.log 2>&1
  echo "lean $f exit $?" | tee -a $O/exits.txt
  rm -rf $S/v-$f
done

# PROVE=smallest: the smallest honest statement (fewest circuit bytes); PROVE=L3: the circuit table's level 3 alone, the smallest
# of V*'s statements Lean lists. Proved --zk at a full draw and verified by Lean.
if [ "${PROVE:-smallest}" = L3 ]; then
  SALTS=$S/acc-salts-l3.bin st level3 $H $S/v-l3
  $P lean --bin $LEAN_BIN --dir $S/v-l3 --inner-circuit $IC --inner-public $IP --out $LJ --label level3 > $E/lean-level3.log 2>&1
  echo "lean level3 exit $?" | tee -a $O/exits.txt
  C=$S/v-l3/L3
else
  C=$(cd $S/v-honest && for d in $(find . -name circuit.txt -printf '%h\n'); do echo "$(stat -c %s $d/circuit.txt) $d"; done | sort -n | head -1 | cut -d' ' -f2)
  C=$S/v-honest/${C#./}
fi
D=$E/prove; mkdir -p $D; echo "$C" > $D/which.txt
free_port() { python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
PIN=$(sha512sum $C/circuit.txt | cut -d' ' -f1) PORT=$(free_port)
$B serve --zk --listen 127.0.0.1:$PORT --out $D/sessions --instances $C/pub.bin --circuit $C/circuit.txt --pin $PIN --operator loopback > $D/serve.log 2>&1 &
SP=$!
for i in $(seq 1800); do grep -q SERVING $D/serve.log && break; kill -0 $SP 2>/dev/null || break; sleep 1; done
/usr/bin/time -v -o $D/time-prove.txt $B prove --zk --verifier 127.0.0.1:$PORT --instances $C/inst.bin --circuit $C/circuit.txt --warm 0 --runs 1 \
  > $D/prove.txt 2> $D/prove.err
echo "prove exit $?" | tee -a $O/exits.txt
kill $SP; wait $SP 2>/dev/null
SESS=$(find $D/sessions -name session.json -printf '%h\n' | sort | head -1)
LV=(--zk --statement verity/flock-circuit --circuit $C/circuit.txt --public $C/pub.bin --partition $C/partition.json --program $C/program.json
    --registered $C/registered.json)
$LEAN_BIN verify "${LV[@]}" --session $SESS > $D/lean.log 2>&1; echo "lean verify exit $?" | tee -a $O/exits.txt
if [ -f $C/rec-bindings.json ]; then
  for tag in hide plain; do
    $LEAN_BIN verify "${LV[@]}" --rec-inner $E/rec-inner-$tag.json --rec-part 3 --rec-bindings $C/rec-bindings.json --rec-rows $C/rec-rows.json \
      --session $SESS > $D/lean-rec-inner-$tag.log 2>&1
    echo "lean verify --rec-inner $tag exit $?" | tee -a $O/exits.txt
  done
fi
$LEAN_BIN verify "${LV[@]}" --rec-inner-session $H/inner/session --rec-inner-transcript $H/inner/public.json --session $SESS > $D/lean-transcript.log 2>&1
echo "lean verify --rec-inner-transcript exit $?" | tee -a $O/exits.txt
$B replay --zk --instances $C/pub.bin --circuit $C/circuit.txt --sessions $SESS > $D/replay.txt 2>&1; echo "replay exit $?" | tee -a $O/exits.txt
cp $C/registered.json $C/partition.json $D/; head -1 $C/circuit.txt > $D/unit.txt; du -sb $C/* > $D/sizes.txt
[ -d $S/v-honest ] && (cd $S/v-honest && find . -name circuit.txt -printf '%h\n' | LC_ALL=C sort | while read d; do echo "$d $(head -1 $d/circuit.txt | cut -c1-120) $(stat -c %s $d/circuit.txt)"; done) > $E/statements.txt
cat $O/exits.txt
