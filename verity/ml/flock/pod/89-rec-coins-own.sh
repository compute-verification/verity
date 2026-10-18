#!/usr/bin/env bash
# VS4: the own-row plain-leaf rec-coins fixture (RoPE d64, n=64), the deployed inner flow's statement (pod 80's
# rec_inner.py stage, own rows, the full stratified draw), for test_lean_rec_coins and test_lean_rec_inner_streams once
# the inner statement is read at Flock's statement check (RecSetup.innerScope). Pod 80's inner side with --plain-leaves and
# the draw recorded (rec_live --draw), then V*'s side over it on the CPU (rec_vstage points, InnerFold.lean's fold, the
# running sums' salts and level 3), the tree assembled under $S/fixture (u/stage the inner statement; inner-honest the
# session's records and session; fold-honest.json, points-honest.json, acc-salts.bin; L3/ pub.bin and registered.json; and
# outer-L3/ reused from the old shared-row fixture, whose session is refused at S2/R7 whatever the level's circuit, so it
# needs no reproof). The tree goes to the store (research data put --kind fixture/v1), its art id and every test constant in
# out/ (ART:, CIRCUIT_SHA512:, REPS:, L3_ACC_READS:, ...). Run from the verity tree (research run --cwd clone).
# env: N=64  CPUS=32  FLOCK_WORK (as 80-rec-inner.sh)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
# research/uv captured before rec-env.sh rewrites PATH to the job's minimal venv
RESEARCH_BIN=$(command -v research || true)
UV_BIN=$(command -v uv || true)
rcli() { if [ -n "$RESEARCH_BIN" ]; then "$RESEARCH_BIN" "$@"; elif [ -n "$UV_BIN" ]; then "$UV_BIN" run --extra torch-cpu research "$@"; else research "$@"; fi; }
if [ -z "${RI_PINNED:-}" ]; then
  . verity/ml/flock/pod/cpu-slices.sh
  hold_slices "" "${CPUS:-32}" || exit 1
  export RI_PINNED=$SLICE_CPUS RAYON_NUM_THREADS=$(echo $SLICE_CPUS | tr ',' '\n' | wc -l)
  exec taskset -c $SLICE_CPUS bash "$0" "$@"
fi
export FC_COINS=os
# the old shared-row fixture, for its outer-L3 session (fetched before rec-env.sh's PATH change)
OLDFIX=verity/ml/flock/tests/fixtures/rec-coins/16-16-8-n64
rcli data fetch-fixtures verity/ml/flock/tests/fixtures/rec-coins > /tmp/fetch-old.log 2>&1 || { echo "fetch old fixture failed"; tail -20 /tmp/fetch-old.log; }
. verity/ml/flock/pod/rec-env.sh

N=${N:-64}; D=$S/stage; V=$S/v; mkdir -p $V
OUTDIR=${OUT:-$O}
BIN=verity/core/service/flock/.lake/build/bin/flock-verify  # only as a fallback label; $LEAN_BIN is rec-env.sh's

# 1. the inner statement: plain SHA-512 leaves, own rows, the full stratified draw (draw.json)
python3 verity/ml/flock/pod/rec_inner.py stage --n $N --sets $S/sets --out $D --plain-leaves > $O/stage.json 2> $O/stage.err || { tail -20 $O/stage.err; exit 1; }
cat $O/stage.json
PART=(--partition $D/partition.json)

# 2. the inner session: the rec proxy serves the full draw (rec_live --draw), the CPU prover proves every unit
$B statement --instances $D/pub-$N.bin --circuit $D/circuit.txt > $O/statement.txt 2>&1 || { tail -5 $O/statement.txt; exit 1; }
python3 -m verity_flock.rec_live --schedule-for $O/statement.txt > $O/schedule.json || exit 1
python3 -m verity_flock.rec_live --listen 127.0.0.1:0 --out $O/inner --schedule $O/schedule.json --draw $D/draw.json > $O/proxy.log 2>&1 &
PP=$!
ADDR=
for i in $(seq 600); do ADDR=$(awk '/^SERVING/ {print $2}' $O/proxy.log); [ -n "$ADDR" ] && break; kill -0 $PP 2>/dev/null || break; sleep 0.5; done
[ -n "$ADDR" ] || { cat $O/proxy.log; exit 1; }
$B prove --verifier $ADDR --instances $D/inst-$N.bin --circuit $D/circuit.txt --runs 1 --warm 0 > $O/prove.txt 2> $O/prove.err
echo "prove exit $?"
wait $PP; echo "proxy exit $?"
grep -h panicked $O/prove.err | head -3
SESS=$O/inner/session
[ -f $SESS/session.json ] || { tail -20 $O/proxy.log $O/prove.err; exit 1; }
$B replay --instances $D/pub-$N.bin --circuit $D/circuit.txt --sessions $SESS > $O/replay.txt 2>&1; echo "replay exit $?"
python3 -m verity_flock.rec_vstar $O/inner > $O/vstar.json 2> $O/vstar.err; echo "vstar exit $?"
wait $LP
# a sanity check only: Lean's verify on the drawn plain-leaf inner session (not a fixture input)
$LEAN_BIN verify --statement verity/flock-circuit+plain-leaves --circuit $D/circuit.txt --public $D/pub-$N.bin "${PART[@]}" \
  --program $D/program.json --stratified $N --coins os --session $SESS > $O/lean.log 2>&1; echo "lean exit $?"; tail -3 $O/lean.log

# 3. InnerFold.lean as an executable (85-rec-reprice.sh's fold, cached by its sources)
FSHA=$( (echo $LSHA; cat verity/ml/flock/tests/lean/InnerFold.lean) | sha256sum | cut -c1-16)
FOLD_BIN=$L/fold-$FSHA/inner-fold
(
  export PATH=$HOME/.elan/bin:$PATH
  command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
  exec 9>$L/fold.lock; flock 9
  if [ ! -x $FOLD_BIN ]; then
    C=$L/fold-src; rm -rf $C && mkdir -p $C && cp -r verity/core/service/flock/. $C/ && cp verity/ml/flock/tests/lean/InnerFold.lean $C/ \
      && printf '\n[[lean_exe]]\nname = "inner-fold"\nroot = "InnerFold"\n' >> $C/lakefile.toml \
      && (cd $C && lake build inner-fold) && mkdir -p $L/fold-$FSHA && cp $C/.lake/build/bin/inner-fold $FOLD_BIN
  fi
) > $O/build-fold.log 2>&1 || { tail -30 $O/build-fold.log; exit 1; }

# 4. V*'s side over the inner session (CPU): points, fold, the running sums' salts, and level 3
python3 -m verity_flock.rec_vstage points $O/inner $D/circuit.txt $D/pub-$N.bin > $V/points.json 2> $O/points.err || { tail -20 $O/points.err; exit 1; }
$FOLD_BIN --plain-leaves --partition $D/partition.json verity/flock-circuit $D/circuit.txt $D/pub-$N.bin $V/points.json os > $V/fold.json 2> $O/fold.err
rc=$?; echo "fold exit $rc"; [ $rc = 0 ] || { tail -20 $O/fold.err; exit 1; }
# the verifier step draws acc-salts.bin, the chain every statement reads
python3 -m verity_flock.rec_vstage stage $O/inner $V/fold.json $V --points $V/points.json --salts $V/acc-salts.bin --levels none --no-algebra > $O/verifier.json 2> $O/verifier.err || { tail -20 $O/verifier.err; exit 1; }
[ -f $V/acc-salts.bin ] || { tail -20 $O/verifier.err; exit 1; }
# level 3, the smallest level the fixture pins
python3 -m verity_flock.rec_vstage stage $O/inner $V/fold.json $V --points $V/points.json --salts $V/acc-salts.bin --levels 3 --no-algebra > $O/L3.json 2> $O/L3.err || { tail -20 $O/L3.err; exit 1; }
[ -f $V/L3/pub.bin ] || { tail -20 $O/L3.err; exit 1; }

# 5. the fixture tree: u/stage {circuit.txt, pub-N.bin, partition.json}; inner-honest {session, public.json, private.json};
# fold-honest.json, points-honest.json, acc-salts.bin; L3 {pub.bin, registered.json}; outer-L3 reused from the old fixture
FX=$S/fixture/16-16-8-n64; mkdir -p $FX/u/stage $FX/inner-honest $FX/L3
cp $D/circuit.txt $D/pub-$N.bin $D/partition.json $FX/u/stage/
cp -r $SESS $FX/inner-honest/session
cp $O/inner/public.json $O/inner/private.json $FX/inner-honest/
cp $V/points.json $FX/points-honest.json
cp $V/fold.json $FX/fold-honest.json
cp $V/acc-salts.bin $FX/acc-salts.bin
cp $V/L3/pub.bin $V/L3/registered.json $FX/L3/
if [ -d $OLDFIX/outer-L3 ]; then cp -r $OLDFIX/outer-L3 $FX/outer-L3; else echo "NO OLD outer-L3 ($OLDFIX)"; fi
du -sb $FX $FX/* > $O/fixture-sizes.txt; cat $O/fixture-sizes.txt
# a copy to $OUT, uploaded by custody as a fallback if research data put fails on the pod
cp -r $S/fixture $OUTDIR/fixture 2>/dev/null || true

# 6. the test constants, derived from the build (greppable, one per line)
L3_SHA=$(sha512sum $V/L3/circuit.txt | cut -d' ' -f1)
echo "CIRCUIT_SHA512: $L3_SHA" | tee $OUTDIR/constants.txt
python3 - "$V/L3/registered.json" >> $OUTDIR/constants.txt <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))["ports"]
print("L3_ACC_READS:", json.dumps(r["acc"]["reads"]))
print("L3_ACC_OUT_READS:", json.dumps(r["acc_out"]["reads"]))
print("L3_OPENED:", len(r["acc"]["reads"]))
EOF
$LEAN_BIN rec-transcript --rec-inner-transcript $O/inner/public.json --rec-inner-session $SESS > $O/reps.txt 2>&1 || true
echo "REPS_LINE: $(grep -h RECTRANSCRIPT $O/reps.txt | head -1)" | tee -a $OUTDIR/constants.txt
cat $OUTDIR/constants.txt

# 7. the fixture to the store
ART=$( rcli data put --kind fixture/v1 --tree $S/fixture --preserve 2> $O/put.err )
PUTRC=$?
echo "ART: $ART (put rc $PUTRC)" | tee $OUTDIR/art.txt
[ $PUTRC = 0 ] || { echo "research data put failed; $OUT/fixture holds the tree"; tail -20 $O/put.err; }
echo "done"
