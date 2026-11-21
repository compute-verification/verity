#!/usr/bin/env bash
# One block of the served circuit (`served_door.py`) committed, proved and verified by Lean on a CPU node, on
# `credited_door.sh`'s road: Lean's flock-lower-check, flock-tmpl-unit and verity-verify-proof and the CPU flock-circuit
# built once per source; the served C built, the template's first unit cut a block of its own and lowered by the door's
# own steps; `served_door.py step stage` commits it on a STAGED witness (its writers' rows random, not served values);
# placed, its statement built, proved over a live loopback session and its session verified by Lean (`--registered`,
# also on a registered file with a wrong output root), then through the door (`served_door.py step verify`). Each wired
# row reads one output field of its writer whole, from that field's tree.
# TEMPLATE: the block's template's descriptor id.
# research run --on auto --cwd source --mem-gb 96 --cpus 16 --timeout 3h -- bash benchmarks/one_stage/served_door_prove.sh
set -uxo pipefail
TEMPLATE=${TEMPLATE:-'OutCollect_v1{KO=2,FIRST=true}'}
export TU_PINNED=slot RAYON_NUM_THREADS=$(nproc)
export FC_COINS=tree
REPO=$(pwd); O=${OUT:-${RESEARCH_RUN_DIR:-/tmp/served-door-prove}/out}; mkdir -p $O
if [ -z "${FLOCK_WORK:-}" ] && [ ! -w /workspace ]; then
  if [ -d /workspace/jobs ]; then export FLOCK_WORK=/workspace/jobs/pc-read-gadget; else export FLOCK_WORK=/workspace/cache/tmpl-unit; fi
fi
FW=${FLOCK_WORK:-/workspace}; W=$FW/flock-circuit; mkdir -p $W || exit 1
[ -w "$HOME" ] || { export HOME=$FW/home; mkdir -p $HOME; }
S=$FW/scratch/${RESEARCH_RUN_ID:-local-$$}; rm -rf $S; mkdir -p $S || exit 1
L=$FW/cache/tmpl-unit; mkdir -p $L || exit 1
unset CARGO_TARGET_DIR
trap 'rm -rf $S' EXIT
{ hostname; lscpu | grep -E 'Model name|^CPU\(s\)'; free -g | head -2; echo "template=$TEMPLATE"; } > $O/host.txt 2>&1; cat $O/host.txt
timed() { local name=$1; shift; /usr/bin/time -o $O/time-$name.txt -f "TIME $name wall_s=%e maxrss_kb=%M rc=%x" "$@"; }
answered() {  # LOG STATUS WANT LINE: a step gave its own answer: status WANT and a line matching LINE, with no traceback
              # and no Lean panic line (verity.lean_process.panic_line); a refusal's status alone is no answer
  [ "$2" = "$3" ] && grep -qE "$4" "$1" && ! grep -q '^Traceback (most recent call last):' "$1" \
    && python3 -c 'import sys
from verity.lean_process import panic_line
sys.exit(panic_line(open(sys.argv[1], errors="replace").read()) is not None)' "$1"
}

LSHA=$(python3 verity/protocols/proof/flock/prover/pod/lake_closure.py key verity/protocols/proof/flock/verifier $REPO/tests/verity/protocols/proof/flock/prover/lean/TmplUnit.lean)
LEAN_BIN=$L/lean-$LSHA/verity-verify-proof; UNIT_BIN=$L/lean-$LSHA/flock-tmpl-unit; FW_BIN=$L/lean-$LSHA/flock-zk-server; LOW_BIN=$L/lean-$LSHA/flock-lower-check
RT_BIN=$L/lean-$LSHA/flock-read-tmpl; CC_BIN=$L/lean-$LSHA/flock-compose-check
(
  export PATH=$HOME/.elan/bin:$PATH
  command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
  exec 9>$L/lean.lock; flock 9
  if [ ! -x $LEAN_BIN ] || [ ! -x $UNIT_BIN ] || [ ! -x $FW_BIN ] || [ ! -x $LOW_BIN ] || [ ! -x $RT_BIN ] || [ ! -x $CC_BIN ]; then
    rm -rf $L/lean-src && LEAN_SRC=$(python3 verity/protocols/proof/flock/prover/pod/lake_closure.py copy verity/protocols/proof/flock/verifier $L/lean-src) \
      && cp tests/verity/protocols/proof/flock/prover/lean/TmplUnit.lean $LEAN_SRC/ \
      && printf '\n[[lean_exe]]\nname = "flock-tmpl-unit"\nroot = "TmplUnit"\n' >> $LEAN_SRC/lakefile.toml \
      && (cd $LEAN_SRC && lake build verity-verify-proof flock-tmpl-unit flock-zk-server flock-lower-check flock-read-tmpl flock-compose-check) && mkdir -p $L/lean-$LSHA \
      && cp $LEAN_SRC/.lake/build/bin/verity-verify-proof $LEAN_BIN && cp $LEAN_SRC/.lake/build/bin/flock-tmpl-unit $UNIT_BIN \
      && cp $LEAN_SRC/.lake/build/bin/flock-zk-server $FW_BIN && cp $LEAN_SRC/.lake/build/bin/flock-lower-check $LOW_BIN \
      && cp $LEAN_SRC/.lake/build/bin/flock-read-tmpl $RT_BIN && cp $LEAN_SRC/.lake/build/bin/flock-compose-check $CC_BIN
  fi
) > $O/build-lean.log 2>&1 &
LP=$!
CHALLENGER_REPO=$REPO CHALLENGER_WORK=$S/challenger-work
. verity/protocols/proof/flock/prover/pod/challenger.sh
challenger_build

KEY=$(cat verity/protocols/proof/flock/prover/pod/20-gpu-link.sh verity/protocols/proof/flock/prover/pod/60-circuit.sh verity/protocols/proof/flock/prover/*.patch verity/protocols/proof/flock/prover/cuda_*_patch.py \
  $(find verity/protocols/proof/flock/prover/cuda verity/protocols/proof/flock/prover/live -type f | LC_ALL=C sort) | sha256sum | cut -c1-16)
BIN=$W/bin-$KEY-g0; F=$FW/flock-gpu-link/flock
exec 8>$W/build.lock; flock 8
if [ -x $BIN/flock-circuit ] && [[ $(ldd $BIN/flock-circuit 2>&1) == *"not found"* ]]; then
  rm -rf $BIN || { echo "$BIN: built against another glibc, and this job can't remove it" >&2; exit 1; }
fi
if [ ! -x $BIN/flock-circuit ]; then
  RESEARCH_RUN_DIR=$O/build FLOCK_WORK=$FW MODE=build GPU=0 bash verity/protocols/proof/flock/prover/pod/60-circuit.sh > $O/build.log 2>&1 || { tail -30 $O/build.log; exit 1; }
  mkdir -p $BIN && cp $F/target/release/flock-circuit $BIN/flock-circuit
fi
exec 8>&-
B=$BIN/flock-circuit; sha256sum $B | tee $O/binary.sha256
UV=$(command -v uv || echo $HOME/.local/bin/uv)
[ -x $UV ] || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; UV=$HOME/.local/bin/uv; }
[ -x $W/py/bin/python ] || { $UV venv -q --python 3.12 $W/py && $UV pip install -q --python $W/py/bin/python numpy blake3; } || exit 1
export PATH=$W/py/bin:$PATH PYTHONPATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/protocols/proof/flock/prover/python:$REPO/adapters/vllm:$REPO/adapters/torch
CHALLENGER_PY=python3
challenger_built || exit 1

$UV sync -q --all-packages --extra torch-cpu || exit 4
wait $LP; [ -x $LEAN_BIN ] && [ -x $UNIT_BIN ] && [ -x $FW_BIN ] && [ -x $LOW_BIN ] && [ -x $RT_BIN ] && [ -x $CC_BIN ] || { tail -20 $O/build-lean.log; exit 1; }
export LEAN_ABORT_ON_PANIC=1  # settings.py's ABORT_ON_PANIC, for every Lean run below and every process they start
SD="$UV run -q python benchmarks/one_stage/served_door.py"
if [ -n "${AUDIT_RULE:-}" ]; then  # the staged window on the audit's road: records_audit.py audit on a local challenger
  WIN="$SD window --tree ."; [ -n "${WINDOW_ARGS:-}" ] && WIN="$UV run -q python $WINDOW_ARGS"  # another window's builder
  ( unset PYTHONPATH; timed window $WIN --out $S/window ) > $O/window.txt 2>&1
  r=$?; cat $O/time-window.txt; grep -h '^STEP' $O/window.txt | cut -c1-3000; [ $r = 0 ] || { tail -c 4000 $O/window.txt; exit 5; }
  cp $S/window/registration.json $S/window/circuit.json $O/
  git -C $REPO rev-list --first-parent HEAD > $S/main.txt 2>/dev/null || git -C $REPO rev-parse HEAD > $S/main.txt || echo 00 > $S/main.txt
  M0=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/protocols/proof/flock/prover/python:$REPO/adapters/vllm
  ( unset PYTHONPATH FC_COINS TU_PINNED; export PATH=$(dirname $RT_BIN):$PATH; RUST_MIN_STACK=$((64 << 20)) timed audit $UV run -q python benchmarks/prove_path/records_audit.py audit \
      --window $S/window --sampling-design "$AUDIT_RULE" --circuit $(sha512sum $S/window/circuit.json | cut -c1-128) \
      --run $(python3 -c 'print("ab" * 64)') --challenger local --deployed-key --registered $S/window/pins.json \
      --registered-digest $(sha512sum $S/window/pins.json | cut -c1-128) --registered-digest-source stand-in --main $S/main.txt --m0-pythonpath $M0 \
      --bin $B --lean $LEAN_BIN --zk-server $FW_BIN --lower $LOW_BIN --tmpl-unit-bin $UNIT_BIN --ports ${AUDIT_PORTS:-64} ${AUDIT_FLAGS:-} --out $S/audit ) > $O/audit.log 2> $O/audit.err
  r=$?; echo "audit rc $r"; cat $O/time-audit.txt; grep -h '^RECORDS\|^REFUSED\|^DRAW\|^MEMBER\|^VERDICT\|^CHECK\|^UNLOWERED\|^A5' $O/audit.log | cut -c1-3000
  tail -c 3000 $O/audit.err
  cp -r $S/audit/summary.json $S/audit/window/summary.json $S/audit/window/case-v2.json $O/ 2>/dev/null
  mkdir -p $O/audit-logs && find $S/audit -name '*.log' -size -20M -exec cp --parents {} $O/audit-logs \; 2>/dev/null
  exit $r
fi
( unset PYTHONPATH; timed build $SD build --tree . --out $S ) > $O/build-C.txt 2>&1; cat $O/time-build.txt; grep -h '^STEP' $O/build-C.txt | cut -c1-600
for s in lower stage; do
  ( unset PYTHONPATH; timed $s $SD step $s --dir $S --template "$TEMPLATE" --lower $LOW_BIN ) > $O/$s.txt 2>&1
  r=$?; cat $O/time-$s.txt; grep -h '^STEP\|STAGE-REFUSED' $O/$s.txt | cut -c1-4000
  [ $r = 0 ] || { tail -c 4000 $O/$s.txt; exit 5; }
done
U0=$(python3 -c 'import json,sys; print(json.loads([l for l in open(sys.argv[1]) if l.startswith("STEP ")][-1][5:])["unit"])' $O/stage.txt)
D=$S/stage-$U0; cp $D/stage.json $D/registered*.json $D/draw.json $D/partition.json $O/
du -b $D/inst-1.bin $D/pub-1.bin $D/circuit.txt | tee $O/stage-sizes.txt
rm -rf $CHALLENGER_WORK/challenger
timed tmpl $UNIT_BIN $D/unit.tmpl --layout iv/hybrid/4 --out $D/unit.placed > $O/tmpl.txt 2>&1 || { tail -40 $O/tmpl.txt; cat $O/time-tmpl.txt; exit 6; }
cat $O/time-tmpl.txt; tail -3 $O/tmpl.txt | cut -c1-600
PIN=$(sha512sum $D/circuit.txt | cut -d' ' -f1)
UA=(--circuit $D/circuit.txt --tmpl-unit $D/unit.placed --tables $D/tables)
timed statement $B statement --zk "${UA[@]}" --instances $D/pub-1.bin > $O/statement.txt 2>&1
r=$?; echo "statement rc $r"; tail -3 $O/statement.txt | cut -c1-1500; cat $O/time-statement.txt
[ $r = 0 ] || exit 8
challenger_start $O/challenger || exit 1
PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
timed serve $B serve --zk --listen 127.0.0.1:$PORT --out $S/sessions --instances $D/pub-1.bin "${UA[@]}" --pin $PIN --operator tmpl-unit \
  --draw subset:1 --sessions 1 "${TREE_FROM[@]}" > $O/serve.log 2>&1 &
SP=$!
for i in $(seq 3600); do grep -q SERVING $O/serve.log && break; kill -0 $SP 2>/dev/null || break; sleep 1; done
timed prove $B prove --zk --verifier 127.0.0.1:$PORT --instances $D/inst-1.bin "${UA[@]}" --runs 1 --warm 0 > $O/prove.txt 2> $O/prove.err
echo "prove rc $?"; cat $O/time-prove.txt; grep -h '^LIVE' $O/prove.txt | cut -c1-1500; tail -5 $O/prove.err | cut -c1-600
for i in $(seq 1200); do kill -0 $SP 2>/dev/null || break; sleep 1; done
kill $SP 2>/dev/null; wait $SP; cat $O/time-serve.txt; grep -h 'accepted\|VERDICT\|REJECT' $O/serve.log | tail -3 | cut -c1-1500
challenger_end $O/challenger; echo "challenger exit $?"
SESS=$(dirname $(ls -1 $S/sessions/*/session.json | LC_ALL=C sort | tail -1)) || { tail -20 $O/serve.log; exit 7; }
du -sb $SESS | tee $O/proof-size.txt
challenger_audit $O/challenger $SESS
bad=0
for r in registered registered-wrong-out; do
  timed lean-$r $LEAN_BIN verify --zk --statement verity/flock-circuit/tmpl --circuit $D/circuit.txt --public $D/pub-1.bin --tables $D/tables \
    --partition $D/partition.json --program $D/program.json --registered $D/$r.json --session $SESS > $O/lean-$r.log 2>&1
  lr=$?; echo "$r lean rc $lr"; cat $O/time-lean-$r.txt; grep -h '^VERDICT' $O/lean-$r.log | head -1 | cut -c1-2500; tail -3 $O/lean-$r.log | cut -c1-600
  if [ $r = registered ]; then want=0 accepted=true; else want=1 accepted=false; fi
  answered $O/lean-$r.log $lr $want "^VERDICT .*\"accepted\":$accepted[,}]" \
    || { echo "ERROR lean-$r: verity-verify-proof verify ended with status $lr and not its own VERDICT accepted=$accepted"; bad=1; }
done
( unset PYTHONPATH; timed door $SD step verify --dir $S --template "$TEMPLATE" --unit $U0 --lower $LOW_BIN --lean $LEAN_BIN --session $SESS ) > $O/door.txt 2>&1
dr=$?; echo "door rc $dr"; cat $O/time-door.txt; grep -h '^STEP' $O/door.txt | cut -c1-3000; tail -c 2000 $O/door.txt | grep -v '^STEP'
answered $O/door.txt $dr 0 '^STEP .*"refusal": null' \
  || { echo "ERROR door: served_door.py step verify ended with status $dr and not its STEP line with no refusal"; bad=1; }
exit $bad
