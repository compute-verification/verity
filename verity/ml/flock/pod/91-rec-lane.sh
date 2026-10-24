#!/usr/bin/env bash
# rec-lane: V*'s recursion (85-rec-reprice.sh) over a served PoUW H1 lane as its inner statement: the lane statement
# (PouwH1LaneZy_v1 at K, one unit, benchmarks/pouw/h1_lanes/served.py) staged from a drawn unit's openings with plain
# SHA-512 leaves (FLOCK_PLAIN_LEAVES=1, the recursive session's inner statement), then 85's steps on it. One job
# (research run --cwd source):
#   STEP=lane    the openings (SMOKE=K: served.py smoke, a stand-in window at K, the harness's own check; else OPENED=F, an
#                extract's private openings, copied 0600 into $FLOCK_WORK/lane-$TAG, never out/), served.py prepare of lane
#                LANE with plain leaves and its own rows (H1S_OWN_ROWS=1: no shared rows and no registered reads, the only
#                inner form the Lean verifier covers ZK off, ProvedScope.sharedRowsOk), and data/$TAG/stage filled as 85's
#                STEP=build leaves it: circuit.txt (META: plain leaves, the lane's partition and Program), inst-1.bin,
#                pub-1.bin, partition.json and program.json (the verifier's), draw.json (rec_live.full_draw over the lane
#                Program: its one unit) and own-partition (85 keeps them). The window binding, outside Lean: served.py bind
#                holds pub-1.bin's row commitments to the window's registered roots (each opened at the row the lane Program
#                reads) and must refuse a copy with one bit of zy's b || c off
#   STEP=none: no lane step (data/$TAG/stage as an earlier job left it)
#   STEPS="build inner icheck vstage vcap oprove oreplay lean" (after lane): 85's, each as
#                `85-rec-reprice.sh STEP=s N=1 STAGE=0`, its wall time and peak RSS in out/time-step-s.txt
# env: FLOCK_WORK (85's)  TAG=lane  LANE=0  SMOKE=  OPENED=  PREP_JOBS=1  INNER_LOOPBACK=0 (no hm96 staging)  INNER_LEAN=1
#      (and 85's: GPU, WARM, RUNS, NEG_INNER, ..)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
REPO=$(pwd); O=${RESEARCH_RUN_DIR:-/tmp/rec-lane-job}/out; mkdir -p $O
FW=${FLOCK_WORK:?FLOCK_WORK}; TAG=${TAG:-lane}; DD=$FW/data/$TAG; LANE=${LANE:-0}; W=$FW/flock-circuit
LW=$FW/lane-$TAG; mkdir -p $DD/stage $W && install -d -m 700 $LW || exit 1
export TAG N=1 STAGE=0 FLOCK_WORK=$FW INNER_LOOPBACK=${INNER_LOOPBACK:-0} INNER_LEAN=${INNER_LEAN:-1}
UV=$(command -v uv || echo $HOME/.local/bin/uv)
[ -x $UV ] || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; UV=$HOME/.local/bin/uv; }
[ -x $W/py/bin/python ] || { $UV venv -q --python 3.12 $W/py && $UV pip install -q --python $W/py/bin/python numpy blake3; } || exit 1
PY=$W/py/bin/python
H=$REPO/benchmarks/pouw/h1_lanes
step() {   # step NAME CMD...: CMD's wall time and peak RSS in out/time-step-NAME.txt, its exit on stdout
  /usr/bin/time -v -o $O/time-step-$1.txt "${@:2}"; local rc=$?
  echo "step $1 exit $rc $(grep -E 'Elapsed|Maximum resident' $O/time-step-$1.txt | sed 's/^\s*//' | tr '\n' ' ')"; return $rc
}

lane() {
  mkdir -p $O/lane
  if [ -n "${SMOKE:-}" ]; then
    PYTHONPATH=$REPO:$REPO/verity/ml/flock/python $PY $H/served.py smoke --k $SMOKE --out $LW/opened.json \
      > $O/lane/smoke.out 2> $O/lane/smoke.err || { tail -5 $O/lane/smoke.err; return 1; }
  else
    install -m 600 "${OPENED:?OPENED or SMOKE}" $LW/opened.json || return 1
  fi
  $PY -c "import json,sys; d=json.load(open(sys.argv[1])); print('openings', d['window'][:16], 'unit', d['unit']['unit'], 'K', d['k'], 'stand-in', bool(d['unit'].get('stand_in')))" \
    $LW/opened.json | tee $O/lane/openings.txt
  rm -rf $LW/work
  H1S_OWN_ROWS=1 FLOCK_PLAIN_LEAVES=1 H1_TMPL= PYTHONPATH=$REPO:$REPO/verity/ml/flock/python /usr/bin/time -v -o $O/lane/time-prepare.txt \
    $PY $H/served.py prepare --opened $LW/opened.json --lanes $LANE --skip tampered-zy,other-call-a --jobs ${PREP_JOBS:-1} \
    --work $LW/work --name rec-lane-$TAG > $O/lane/prepare.out 2> $O/lane/prepare.err || { tail -20 $O/lane/prepare.err; return 1; }
  cat $O/lane/prepare.out
  cp $LW/work/prepare.json $O/lane/
  local s=$LW/work/st/lane$LANE v=$LW/work/verifier D=$DD/stage
  rm -f $D/*
  ln -f "$(readlink -f $s/circuit.txt)" $D/circuit.txt 2>/dev/null || cp "$(readlink -f $s/circuit.txt)" $D/circuit.txt
  cp $s/inst.bin $D/inst-1.bin && cp $v/lane$LANE.pub.bin $D/pub-1.bin && cp $v/lane.partition.json $D/partition.json \
    && cp $v/lane.program.json $D/program.json || return 1
  cmp -s $s/pub.bin $D/pub-1.bin; echo "the prover's pub.bin is the verifier's: $([ $? = 0 ] && echo yes || echo no)"
  # the full draw over the lane Program (rebuilt as served.py prepare builds it), refused unless META binds exactly the
  # partition and Program the verifier wrote
  PYTHONPATH=$REPO:$REPO/verity/ml/flock/python $PY - $D $($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['k'])" $LW/opened.json) <<'EOF' || return 1
import json, sys
from pathlib import Path
sys.path.insert(0, "benchmarks/pouw/h1_lanes")
import served as S
from verity.core.protocols.sampled_proofs.one_stage import partition as P
from verity_flock import circuit as FC
from verity_flock import rec_live as RL

d, k = Path(sys.argv[1]), int(sys.argv[2])
b = S.L.bound("H1LanesLaneStatement", S.L.lane_word(S.Z.lane(S.L.tile(k))))
part = json.loads((d / "partition.json").read_text())
if P.digest(part) != P.digest(b.part):
    raise SystemExit("the verifier's partition is not the lane Program's")
with open(d / "circuit.txt") as f:
    f.readline()
    meta = json.loads(f.readline()[5:])
if (meta["partition"]["digest"], meta["program_sha512"]) != (P.digest(part), part["program"]):
    raise SystemExit(f"META binds partition {meta['partition']['digest']} over {meta['program_sha512']}, not the lane's")
if meta.get("leaf_scheme") != FC.LEAF_SCHEME_PLAIN:
    raise SystemExit(f"META's leaf scheme is {meta.get('leaf_scheme')}, not plain SHA-512 (FLOCK_PLAIN_LEAVES=1)")
draw = RL.full_draw(b.program, b.query)
if draw["population"] != 1:
    raise SystemExit(f"the lane Program's population is {draw['population']}, not one unit")
(d / "draw.json").write_text(json.dumps(draw))
print(json.dumps({"partition": P.digest(part), "program": part["program"], "population": draw["population"],
                  "leaf_scheme": meta["leaf_scheme"]["id"]}))
EOF
  # the window binding: the honest public file holds, and one with zy's b || c one bit off is refused
  PYTHONPATH=$REPO:$REPO/verity/ml/flock/python $PY $H/served.py bind --pub $D/pub-1.bin --work $LW/work --tag lane$LANE \
    | tee $O/lane/bind.txt; [ ${PIPESTATUS[0]} = 0 ] || { echo "the window binding fails"; return 1; }
  $PY - $D/pub-1.bin $D/circuit.txt $LW/pub-forged.bin <<'EOF' || return 1
import json, sys
raw = open(sys.argv[1], "rb").read()
with open(sys.argv[2]) as f:
    f.readline()
    names = [p[0] for p in json.loads(f.readline()[5:])["ports"]]
at = raw.index(b"\n") + 1 + 128 * names.index("zy")
open(sys.argv[3], "wb").write(raw[:at] + bytes([raw[at] ^ 1]) + raw[at + 1:])
EOF
  PYTHONPATH=$REPO:$REPO/verity/ml/flock/python $PY $H/served.py bind --pub $LW/pub-forged.bin --work $LW/work --tag lane$LANE \
    > $O/lane/bind-forged.txt; rc=$?; cat $O/lane/bind-forged.txt
  [ $rc = 1 ] || { echo "the window binding accepts a forged zy row (exit $rc)"; return 1; }
  touch $D/own-partition
  du -sb $D/* | tee $O/lane/sizes.txt
  sha256sum $D/* > $O/lane/stage-sha256.txt
}

if [ "${STEP:-lane}" = lane ]; then
  t0=$SECONDS; lane; rc=$?; echo "step lane exit $rc $((SECONDS - t0)) s (prepare: out/lane/time-prepare.txt)"; [ $rc = 0 ] || exit 1
fi
for s in ${STEPS-build inner icheck vstage vcap oprove oreplay lean}; do
  step $s env RESEARCH_RUN_DIR=${RESEARCH_RUN_DIR:-/tmp/rec-lane-job} bash verity/ml/flock/pod/85-rec-reprice.sh STEP=$s \
    > $O/step-$s.log 2>&1
  rc=$?; tail -3 $O/step-$s.log
  [ $rc = 0 ] || { echo "rec-lane: STEP=$s failed (out/step-$s.log)"; exit 1; }
done
