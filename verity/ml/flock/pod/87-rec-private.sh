#!/usr/bin/env bash
# rec-private (lane rec-v0): a hidden netlist as the recursion's inner statement. The universal unit UniversalUnit_v1{CLS}
# (verity.ml.private_circuits.universal) runs a random netlist padded to the class, its op/src/out one row every instance
# opens (the registered program: rec_inner.program_unit, so three regions), x a row per instance, staged with plain leaves
# and salts from the OS (rec_inner.py universal); its inner session is proved on the CPU (M0, FC_COINS=os) against the rec
# proxy, whose challenger commits to every coin first (rec_live: coins.json), opening a
# recursive session on the firewall's state
# (rec_live --firewall, the parts V* will prove in order), the firewall's contract (Lean flock-firewall) checks what the
# proxy let out, V* is staged over it (rec_vstage points, InnerFold.lean, rec_vstage stage under the class's cap), and each
# of V*'s statements is proved with --zk behind the outer firewall (FC_FIREWALL=1: the send check, released by the firewall
# as part NAME of the recursive session, --firewall-state), each release decided again by Lean's core-firewall
# (verity/firewall-release/v0) on the record before it, and each session verified by Lean's flock-verify verify --zk and
# replayed by the live build. Each outer session is served with the verifier's draw of every unit (rec-private-outer.sh),
# and each of V*'s statements reads registered values, a shared-row public file at hidden outputs (--registered); a form Lean
# refuses is "Lean refused: form not covered" in summary.json, a refusal, and its lean_of_record accepts the run only when
# Lean accepted every part.
# The controls, each refused where it should be:
#   faulty       a faulty unit (rec_inner.py faulty: instance 0's output row one bit off in the prover's file, the prover
#                checking nothing), proved against the registered statement: V*'s algebra over its claims (alg parts, outer
#                sessions sent though the witness fails, FC_ZK_SELF_CHECK=skip)
#   outside      a description that is not a circuit (rec_inner.py universal --outside: a gate reading a wire computed
#                after it), registered as its own statement: a member of the class, so nothing refuses it, which the run
#                shows (its algebra proved and verified as an honest statement's, outside-alg-p*)
#   other        another netlist of the class (the registered one with an unread no-op gate changed: same outputs), proved
#                against the registered statement: V*'s algebra over the inner's claims at the registered statement's fold
#                (alg parts; outer sessions sent though the witness fails, FC_ZK_SELF_CHECK=skip)
#   other-coins  the honest inner transcript against coins other than the challenger committed to (rec_inner.py forge --what
#                challenge on a copy): the firewall's contract on the run's transcript with that coin swapped
#                (rec_private_coins.py), and V*'s algebra over the forged session (alg parts, as other's)
#   challenger   firewall_agree.py --cases: challengers whose coins do not open their commitments, which both the firewall
#                and its contract refuse, beside the run's own session
#   made-up      --wired's cheater (universal.run_wired: layer 1 reads a made-up input): the verifier's Linking check on
#                the inner statement's public file, live (statement --links, before any session) and in V*'s fold
#                (InnerFold.lean --program, the setup Lean's verify runs) over a session recorded under no links; the
#                honest wired statement passes both, and neither passes the fold without the verifier's program
#   flipped-top  rec_inner.py forge --what top: V*'s reference (rec_vstar) refuses it by name; staged by a prover that skips
#                its own top checks (rec_inner.py flipped), the level-1 statement, refused by Lean's verify --zk
#   forged-top   rec_vstage --forge top: an opening that climbs to a node other than its committed top (the last level)
# Run from the verity tree (research run --cwd source). Big files stay under $FLOCK_WORK/data/rec-private/$TAG; out/ gets
# the records and summary.json.
# env: CLS=16,16,8  N=64 (instances, two a block: m = k_log + 5)  NW=8 (wired: per layer, 4 NW in all)  SEED=0  CPUS=32
#      FLOCK_WORK (as 80-rec-inner.sh)  TAG (default CLS-nN)  LEAN_FROM (a cache directory whose lean-<sha>/flock-verify
#      of the same sources is used)  STAGED STAGED_RECORD (an earlier staging, not restaged: 90-rec-stage.sh MODE=fast's DIR
#      with stage/ and other/, or stage_class.py's statement alone, whose N its pub-N.bin names)  WCLS (the wired circuit's
#      class)  CAP_FOLD (an earlier fold of the statement: the cap from it, the other netlist's chain in the background)
#      OUTER_DRAW (rec-private-outer.sh: the verifier's draw, default all)  REC_FAST_STAGE=1 (rec_inner.py universal --fast)
#      CONTROLS=0 (none of faulty, outside, other-coins; outside is never staged beside STAGED)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
if [ -z "${RI_PINNED:-}" ]; then
  . verity/ml/flock/pod/cpu-slices.sh
  hold_slices "" "${CPUS:-32}" || exit 1
  export RI_PINNED=$SLICE_CPUS RAYON_NUM_THREADS=$(echo $SLICE_CPUS | tr ',' '\n' | wc -l)
  exec taskset -c $SLICE_CPUS bash "$0" "$@"
fi
export FC_COINS=os FLOCK_WORK=${FLOCK_WORK:-/workspace/jobs/rec-v0}
LSHA0=$(cd verity/core/service/flock && find . -name .lake -prune -o -type f -print | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)
FW0=$FLOCK_WORK
if [ ! -x $FW0/cache/rec-v0/lean-$LSHA0/flock-verify ]; then
  for c in ${LEAN_FROM:-} /workspace/jobs/*/cache /workspace/jobs/*/cache/*; do
    [ -x $c/lean-$LSHA0/flock-verify ] || continue
    mkdir -p $FW0/cache/rec-v0/lean-$LSHA0 && cp $c/lean-$LSHA0/flock-verify $FW0/cache/rec-v0/lean-$LSHA0/ && break
  done
fi
. verity/ml/flock/pod/rec-env.sh

if [ -n "${STAGED:-}" ] && [ -z "${N:-}" ]; then
  N=$(basename $(ls $STAGED/stage/pub-*.bin $STAGED/pub-*.bin 2>/dev/null | head -1) .bin); N=${N#pub-}
fi
CLS=${CLS:-16,16,8}; N=${N:-64}; NW=${NW:-8}; SEED=${SEED:-0}; TAG=${TAG:-${CLS//,/-}-n$N}
DD=$FW/data/rec-private/$TAG; rm -rf $DD; mkdir -p $DD $O/inner $O/vstar $O/outer || exit 1
command -v research >/dev/null && research keep $DD --owner @proofs --expires 2026-10-23 --priority low --why "rec-private $TAG" > /dev/null 2>&1
PY=verity/ml/flock/pod/rec_inner.py
FWS=$DD/firewall

# Lean's tools beside flock-verify, in one copy of the packages they need, cached by their sources: flock-firewall (the
# firewall's contract), inner-fold (tests/lean/InnerFold.lean, V*'s native side) and core-firewall (the release decision)
TSHA=$( (find verity/core/service/flock verity/core/service/firewall verity/core/protocols/interface verity/core/difftest/lean \
  -name .lake -prune -o -type f -print | LC_ALL=C sort | xargs sha256sum; cat verity/ml/flock/tests/lean/InnerFold.lean) | sha256sum | cut -c1-16)
T=$L/tools-$TSHA
(
  export PATH=$HOME/.elan/bin:$PATH
  command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
  exec 9>$L/tools.lock; flock 9
  [ -x $T/core-firewall ] && [ -x $T/inner-fold ] && [ -x $T/flock-firewall ] && exit 0
  C=$L/tools-src; rm -rf $C && mkdir -p $C/verity/core/service $C/verity/core/protocols $C/verity/core/difftest || exit 1
  cp -r verity/core/service/flock verity/core/service/firewall $C/verity/core/service/ && cp -r verity/core/protocols/interface $C/verity/core/protocols/ \
    && cp -r verity/core/difftest/lean $C/verity/core/difftest/ || exit 1
  find $C -name .lake -prune -exec rm -rf {} +
  [ -d $L/lean-src/.lake/build ] && cp -r $L/lean-src/.lake $C/verity/core/service/flock/
  cp verity/ml/flock/tests/lean/InnerFold.lean $C/verity/core/service/flock/ \
    && printf '\n[[lean_exe]]\nname = "inner-fold"\nroot = "InnerFold"\n' >> $C/verity/core/service/flock/lakefile.toml || exit 1
  (cd $C/verity/core/service/flock && lake build flock-firewall inner-fold) || exit 1
  (cd $C/verity/core/service/firewall && lake build core-firewall) || exit 1
  mkdir -p $T && cp $C/verity/core/service/flock/.lake/build/bin/inner-fold $C/verity/core/service/flock/.lake/build/bin/flock-firewall \
    $C/verity/core/service/firewall/.lake/build/bin/core-firewall $T/
) > $O/build-tools.log 2>&1 &
TP=$!

. verity/ml/flock/pod/rec-private-outer.sh
pub_of() { ls $1/pub-*.bin | head -1; }

# stage: the universal unit's statement and the other netlist's (u/), the wired circuit's honest and cheating ones (w/)
if [ -n "${STAGED:-}" ] && [ -s $STAGED/circuit.txt ]; then
  # an earlier staging of the statement alone (benchmarks/private_circuit/stage.sh STAGE_DIR's G-N_IN-N_OUT: stage_class.py,
  # the unit itself the program, no program row), its files linked into u/stage beside the partition object its circuit
  # binds; no other netlist is staged under its salts, so there is no other chain; STAGED_RECORD: that run's row.json
  mkdir -p $DD/u/stage && ln -s $STAGED/* $DD/u/stage/ && rm -f $DD/u/stage/partition.json || exit 1
  python3 $PY partition --cls $CLS --circuit $STAGED/circuit.txt --out $DD/u/stage/partition.json > $O/stage-partition.json || exit 1
  [ -n "${STAGED_RECORD:-}" ] && cp $STAGED_RECORD $O/stage-universal.json
elif [ -n "${STAGED:-}" ]; then
  # an earlier staging of the class's two statements (90-rec-stage.sh MODE=fast's DIR, with stage/ and other/), linked, not
  # restaged; STAGED_RECORD: that run's summary.json, whose universal record is this run's stage-universal.json
  [ -s $STAGED/stage/circuit.txt ] && [ -s $STAGED/other/circuit.txt ] || { echo "no staging under $STAGED"; exit 1; }
  mkdir -p $DD/u && ln -s $STAGED/stage $DD/u/stage && ln -s $STAGED/other $DD/u/other || exit 1
  [ -n "${STAGED_RECORD:-}" ] && python3 -c "import json, sys; print(json.dumps(json.load(open(sys.argv[1]))['universal']))" \
    $STAGED_RECORD > $O/stage-universal.json
else
  /usr/bin/time -v -o $O/time-stage-universal.txt python3 $PY universal --cls $CLS --n $N --seed $SEED --out $DD/u > $O/stage-universal.json 2> $O/stage-universal.err \
    || { tail -20 $O/stage-universal.err; exit 1; }
fi
# in the background: WCLS, the wired circuit's class (default CLS), staged by today's path (four layers of the unit), and the
# description outside the class (x/), each waited for where it is first read
/usr/bin/time -v -o $O/time-stage-wired.txt python3 $PY wired --cls ${WCLS:-$CLS} --n $NW --seed $SEED --out $DD/w > $O/stage-wired.json 2> $O/stage-wired.err &
WS=$!
XS=
if [ "${CONTROLS:-1}" = 1 ] && [ -z "${STAGED:-}" ]; then
  ( /usr/bin/time -v -o $O/time-stage-outside.txt python3 $PY universal --cls $CLS --n $N --seed $SEED --out $DD/x --outside \
      > $O/stage-outside.json 2> $O/stage-outside.err; echo $? > $DD/stage-outside.exit ) &
  XS=$!
fi
cp $DD/u/stage/partition.json $O/
U=$DD/u/stage; PUB=$(pub_of $U)

# inner NAME STMT INST [firewall|-] [LINKS]: the inner session of statement STMT (its circuit and public file: the schedule of
# its public shape, Linking checked on that file under the verifier's LINKS when the circuit declares links) proved from the
# prover's file INST against the rec proxy, into $DD/inner-NAME; the live build's replay of it, and V*'s points from its coins
# (rec_vstage points) into $DD/points-NAME.json
inner() {
  local name=$1 st=$2 inst=$3 fw=${4:--} D=$O/inner/$1 I=$DD/inner-$1 pub=$(pub_of $2) FWA=() LKA=() ADDR= PP
  [ -n "${5:-}" ] && LKA=(--links $5)
  mkdir -p $D; rm -rf $I
  $B statement --instances $pub --circuit $st/circuit.txt "${LKA[@]}" > $D/statement.txt 2>&1 || { tail -5 $D/statement.txt; return 1; }
  python3 -m verity_flock.rec_live --schedule-for $D/statement.txt > $D/schedule.json || return 1
  [ $fw = firewall ] && FWA=(--firewall $FWS --parts $PARTS)
  python3 -m verity_flock.rec_live --listen 127.0.0.1:0 --out $I --schedule $D/schedule.json "${FWA[@]}" > $D/proxy.log 2>&1 &
  PP=$!
  for i in $(seq 600); do ADDR=$(awk '/^SERVING/ {print $2}' $D/proxy.log); [ -n "$ADDR" ] && break; kill -0 $PP 2>/dev/null || break; sleep 0.5; done
  [ -n "$ADDR" ] || { cat $D/proxy.log; return 1; }
  /usr/bin/time -v -o $D/time-prove.txt $B prove --verifier $ADDR --instances $inst --circuit $st/circuit.txt --runs 1 --warm 0 \
    > $D/prove.txt 2> $D/prove.err
  echo "$name inner prove exit $?"
  wait $PP; echo "$name proxy exit $?"
  [ -f $I/session/session.json ] || { tail -20 $D/proxy.log $D/prove.err; return 1; }
  $B replay --instances $pub --circuit $st/circuit.txt "${LKA[@]}" --sessions $I/session > $D/replay.txt 2>&1
  cp $I/public.json $D/ && du -sb $I/session/* $I/*.json > $D/sizes.txt
  python3 -m verity_flock.rec_vstage points $I $st/circuit.txt $pub > $DD/points-$name.json 2> $D/points.err || { tail -5 $D/points.err; return 1; }
}

# fold NAME STMT [--program FILE]: V*'s native side (InnerFold.lean: the extra claims and each rep's comb, from the statement
# and the session's coins) at $DD/points-NAME.json, into $DD/fold-NAME.json; its status
fold() {
  local name=$1 st=$2; shift 2
  /usr/bin/time -v -o $O/vstar/time-fold-$name.txt $T/inner-fold --plain-leaves --partition $st/partition.json "$@" verity/flock-circuit \
    $st/circuit.txt $(pub_of $st) $DD/points-$name.json os > $DD/fold-$name.json 2> $O/vstar/fold-$name.err
  local rc=$?; echo "fold $name exit $rc"; echo $rc > $O/vstar/fold-$name.exit
  return $rc
}

# the other netlist first: its inner session (against the registered statement) and V*'s fold over the registered statement
# give the claims, so the class's cap and the recursive session's parts are known before the honest session opens one
# CAP_FOLD: an earlier fold of this statement (its extra claims are the statement's, so the cap is the same), and the other
# netlist's chain (its inner session, fold and V*'s staging) runs in the background beside the honest one. With no other
# netlist (a staging of the statement alone), the cap is folded from the honest prover file's session outside the firewall
CAPOF=other; CAPINST=$DD/u/other/inst-$N.bin
[ -d $DD/u/other ] || { CAPOF=cap; CAPINST=$U/inst-$N.bin; }
[ -z "${CAP_FOLD:-}" ] && { inner $CAPOF $U $CAPINST || exit 1; }
wait $LP; echo "flock-verify build exit $?"; [ -x $LEAN_BIN ] || { tail -20 $O/build-lean.log; exit 1; }
wait $TP; echo "tools build exit $?"; [ -x $T/inner-fold ] && [ -x $T/core-firewall ] || { tail -30 $O/build-tools.log; exit 1; }
[ -z "${CAP_FOLD:-}" ] && { fold $CAPOF $U || exit 1; }
for i in $(seq 360); do [ -z "${CAP_FOLD:-}" ] || [ -s $CAP_FOLD ] && break; sleep 20; done
python3 - $U/circuit.txt $N ${CAP_FOLD:-$DD/fold-$CAPOF.json} > $DD/cap.json <<'EOF' || exit 1
import json, sys
from pathlib import Path
from verity_flock import rec_shape as RS
cap = RS.Cap.of(Path(sys.argv[1]), int(sys.argv[2]), 2 + len(json.loads(Path(sys.argv[3]).read_text())["extras"]))
print(json.dumps({"cap": cap.spec(), "shape": [cap.shape.m, cap.shape.k_log, cap.shape.claims], "parts": list(RS.outer(cap)),
                  "statements": RS.outer(cap)}))
EOF
cp $DD/cap.json $O/
CAP=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['cap'])" $DD/cap.json)
PARTS=$(python3 -c "import json, sys; print(','.join(json.load(open(sys.argv[1]))['parts']))" $DD/cap.json)
if [ -n "${CAP_FOLD:-}" ] && [ -d $DD/u/other ]; then
  ( inner other $U $DD/u/other/inst-$N.bin && fold other $U \
      && /usr/bin/time -v -o $O/vstar/time-other.txt python3 -m verity_flock.rec_vstage stage $DD/inner-other $DD/fold-other.json $DD/vo \
           --points $DD/points-other.json --salts $DD/vo-acc-salts.bin --levels none --cap $CAP > $O/vstar/other.json 2> $O/vstar/other.err
    echo "other chain exit $?" ) > $O/other-chain.log 2>&1 &
  OC=$!
fi

# the faulty unit's and the outside description's chains in the background, beside the honest one: the inner session, V*'s
# fold and V*'s algebra (--levels none), into $DD/vx and $DD/vu
achain() {
  local name=$1 st=$2 inst=$3 out=$4
  inner $name $st $inst && fold $name $st \
    && /usr/bin/time -v -o $O/vstar/time-$name.txt python3 -m verity_flock.rec_vstage stage $DD/inner-$name $DD/fold-$name.json $out \
         --points $DD/points-$name.json --salts $out-acc-salts.bin --levels none --cap $CAP > $O/vstar/$name.json 2> $O/vstar/$name.err
  echo "$name chain exit $?"
}
XC= UC=
if [ "${CONTROLS:-1}" = 1 ]; then
  ( python3 $PY faulty --stage $U --n $N --out $DD/u/faulty > $O/stage-faulty.json && achain faulty $U $DD/u/faulty/inst-$N.bin $DD/vx ) \
    > $O/faulty-chain.log 2>&1 &
  XC=$!
  if [ -n "$XS" ]; then
    ( until [ -s $DD/stage-outside.exit ]; do sleep 10; done
      [ "$(cat $DD/stage-outside.exit)" = 0 ] || { tail -20 $O/stage-outside.err; exit 1; }
      achain outside $DD/x/stage $DD/x/stage/inst-$N.bin $DD/vu ) > $O/outside-chain.log 2>&1 &
    UC=$!
  fi
fi

# honest: the inner session through the firewall (a recursive session opened with PARTS), the firewall's contract on what it
# let out, the inner verified by Lean as far as it goes (plain leaves: refused until soundness covers them), V*'s fold
inner honest $U $U/inst-$N.bin firewall || exit 1
SID=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['session'])" $DD/inner-honest/firewall.json) || exit 1
echo $SID > $O/inner/honest/firewall-session.txt
python3 verity/ml/flock/verifier/firewall_agree.py --lean $T/flock-firewall --out $O/inner/firewall-contract --sessions $DD/inner-honest \
  > $O/inner/firewall-contract.txt 2>&1; echo "inner firewall contract exit $?"
cp $DD/inner-honest/coins.json $O/inner/honest/
python3 verity/ml/flock/verifier/firewall_agree.py --lean $T/flock-firewall --out $O/inner/firewall-cases --cases \
  > $O/inner/firewall-cases.txt 2>&1; echo "firewall cases exit $?"
python3 verity/ml/flock/pod/rec_private_coins.py --lean $T/flock-firewall --session $DD/inner-honest --out $O/inner/coins-contract \
  > $O/inner/coins-contract.txt 2>&1; echo "coins contract exit $?"
CC=
if [ "${CONTROLS:-1}" = 1 ]; then
  # other-coins: a copy of the honest session with rep 0's challenge coin one bit off, as the session and the coin record have it
  IC=$DD/inner-other-coins; rm -rf $IC; mkdir -p $IC $O/inner/other-coins
  cp -r $DD/inner-honest/session $DD/inner-honest/public.json $DD/inner-honest/private.json $DD/inner-honest/coins.json $IC/ || exit 1
  python3 $PY forge --what challenge --inner $IC > $O/inner/other-coins/forge.json || exit 1
  ( python3 -m verity_flock.rec_vstage points $IC $U/circuit.txt $PUB > $DD/points-other-coins.json 2> $O/inner/other-coins/points.err \
      && fold other-coins $U \
      && /usr/bin/time -v -o $O/vstar/time-other-coins.txt python3 -m verity_flock.rec_vstage stage $IC $DD/fold-other-coins.json $DD/vc \
           --points $DD/points-other-coins.json --salts $DD/vc-acc-salts.bin --levels none --cap $CAP > $O/vstar/other-coins.json 2> $O/vstar/other-coins.err
    echo "other-coins chain exit $?" ) > $O/other-coins-chain.log 2>&1 &
  CC=$!
fi
( /usr/bin/time -v -o $O/inner/honest/time-lean.txt $LEAN_BIN verify --statement verity/flock-circuit+plain-leaves --circuit $U/circuit.txt \
    --public $PUB --partition $U/partition.json --coins os --session $DD/inner-honest/session > $O/inner/honest/lean.log 2> $O/inner/honest/lean.err
  echo "inner lean exit $?" ) &
IL=$!
fold honest $U || exit 1
python3 -m verity_flock.rec_vstar $DD/inner-honest > $O/vstar/vstar-honest.json 2> $O/vstar/vstar-honest.err; echo "vstar honest exit $?"

# V*'s statements: honest (every level and the algebra, under the class's cap), the wrong top, and the other netlist's algebra
V=$DD/v; mkdir -p $V
vstage() {
  local name=$1; shift
  /usr/bin/time -v -o $O/vstar/time-$name.txt python3 -m verity_flock.rec_vstage stage "$@" --cap $CAP > $O/vstar/$name.json 2> $O/vstar/$name.err
  echo "vstage $name exit $?"
}
vstage honest $DD/inner-honest $DD/fold-honest.json $V --points $DD/points-honest.json --salts $V/acc-salts.bin
[ -f $V/acc-salts.bin ] || { tail -20 $O/vstar/honest.err; exit 1; }
vstage forged-top $DD/inner-honest $DD/fold-honest.json $V --points $DD/points-honest.json --salts $V/acc-salts.bin --forge top
[ -z "${CAP_FOLD:-}" ] && [ -d $DD/u/other ] && vstage other $DD/inner-other $DD/fold-other.json $DD/vo --points $DD/points-other.json --salts $DD/vo-acc-salts.bin --levels none
(cd $V && du -sb $(find . -name '*.bin' -o -name circuit.txt | LC_ALL=C sort)) > $O/vstar/sizes.txt

# the flipped top: rec_inner.py forge --what top on a copy of the honest session; V*'s reference and staging refuse it by name,
# and a prover that skips its own checks stages the level it flipped
F=$DD/inner-flipped; mkdir -p $F && cp $DD/inner-honest/public.json $DD/inner-honest/private.json $F/ && ln -s $DD/inner-honest/session $F/session
python3 $PY forge --what top --inner $F > $O/vstar/forge-top.json || exit 1
python3 -m verity_flock.rec_vstar $F > $O/vstar/vstar-flipped.json 2> $O/vstar/vstar-flipped.err; echo "vstar flipped exit $?"
vstage flipped-checked $F $DD/fold-honest.json $DD/vf-checked --points $DD/points-honest.json --salts $V/acc-salts.bin --levels 1 --no-algebra
python3 $PY flipped --inner $F --fold $DD/fold-honest.json --points $DD/points-honest.json --salts $V/acc-salts.bin --out $DD/vf \
  > $O/vstar/flipped.json 2> $O/vstar/flipped.err; echo "flipped stage exit $?"

# the wired circuit: the honest statement's inner session under the verifier's links, V*'s fold with the verifier's program
# (Linking) and without it; the cheater's public file under the verifier's links (the live build's Linking refuses it before
# any session), its inner session under none (a verifier that checks no Linking records it), and V*'s fold of it with the program
wait $WS || { tail -20 $O/stage-wired.err; exit 1; }
W=$DD/w/stage
inner wired $W $(ls $W/inst-*.bin) - $W/links.json || exit 1
fold wired $W --program $W/program.desc
cp $DD/points-wired.json $DD/points-wired-noprogram.json
fold wired-noprogram $W
mkdir -p $O/inner/made-up
$B statement --instances $(pub_of $DD/w/made-up-input) --circuit $DD/w/made-up-input/circuit.txt --links $W/links.json \
  > $O/inner/made-up/own-links.txt 2>&1; echo "made-up statement under the verifier's links exit $?"
inner made-up $DD/w/made-up-input $(ls $DD/w/made-up-input/inst-*.bin) - $DD/w/no-links.json || exit 1
fold made-up $DD/w/made-up-input --program $DD/w/made-up-input/program.desc

# the outer sessions: honest through the firewall in the recursive session's order, then the controls'. The firewall
# releases a part only when its verdict accepts it, and every V* statement's is the refusal of its form, so the recursive
# session stops at its first part: each later one is proved outside it (NAME-direct), for each verifier's verdict on it
for p in ${PARTS//,/ }; do
  case $p in L*) C=$V/$p ;; alg-p*) C=$V/alg/p${p#alg-p} ;; esac
  outer $p $C fw || outer $p-direct $C plain
done
LAST=$(ls -d $V/forged-top/L* | head -1)
outer forged-top-$(basename $LAST) $LAST dev
outer flipped-top-L1 $DD/vf/forged-flipped-top/L1 dev
[ -n "${OC:-}" ] && { wait $OC; tail -3 $O/other-chain.log; }
for C in $(ls -d $DD/vo/alg/p* 2>/dev/null | sort -V); do outer other-alg-$(basename $C) $C dev; done
for c in $XC $CC $UC; do wait $c; done
tail -n2 $O/faulty-chain.log $O/other-coins-chain.log $O/outside-chain.log
for C in $(ls -d $DD/vx/alg/p* 2>/dev/null | sort -V); do outer faulty-alg-$(basename $C) $C dev; done
for C in $(ls -d $DD/vc/alg/p* 2>/dev/null | sort -V); do outer other-coins-alg-$(basename $C) $C dev; done
for C in $(ls -d $DD/vu/alg/p* 2>/dev/null | sort -V); do outer outside-alg-$(basename $C) $C plain; done
cp $FWS/record.jsonl $O/firewall-record.jsonl
wait $IL
bash verity/ml/flock/pod/88-rec-private-public.sh TAG=$TAG > $O/public.log 2>&1; echo "public views exit $?"

python3 verity/ml/flock/pod/rec_private_summary.py $O
true
