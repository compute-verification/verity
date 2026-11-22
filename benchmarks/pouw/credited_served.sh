#!/usr/bin/env bash
# Pearl+'s tile in the served circuit (`credited_served.py`), proved and verified on a CPU node, as `credited_in_c.sh` does
# its C: Lean's flock-lower-check, flock-tmpl-unit and verity-verify-proof and the CPU flock-circuit built once per source;
# `credited_served.py stage` builds WIDE's class 1,1,1 circuit with the tile at layer 0's qkv unit 0, lays the tile, samples
# one block of its core (STEER, default PearlPlusTileCoreWired; TN its columns, default 4, the grain's GEMM unit K · TN words) and stages it twice, OUT/honest and OUT/flipped (one weight
# word flipped under the registration's read); each is placed (flock-tmpl-unit), its statement built, proved over a live
# loopback session (`flock-circuit serve --zk`) and verified by Lean (`verity-verify-proof verify --zk --registered`);
# `credited_served.py door` then puts both through `proofs.verify`.
# research run --on auto --cwd source --wait --mem-gb 64 --cpus 16 --timeout 180m -- bash benchmarks/pouw/credited_served.sh
set -uxo pipefail
STEER=${STEER:-PearlPlusTileCoreWired} TN=${TN:-4} CORE=${CORE:-v3}
export TU_PINNED=slot RAYON_NUM_THREADS=$(nproc)
export FC_COINS=tree
REPO=$(pwd); O=${OUT:-${RESEARCH_RUN_DIR:-/tmp/credited-served}/out}; mkdir -p $O
if [ -z "${FLOCK_WORK:-}" ] && [ ! -w /workspace ]; then
  if [ -d /workspace/jobs ]; then export FLOCK_WORK=/workspace/jobs/pc-read-gadget; else export FLOCK_WORK=/workspace/cache/tmpl-unit; fi
fi
FW=${FLOCK_WORK:-/workspace}; W=$FW/flock-circuit; mkdir -p $W || exit 1
[ -w "$HOME" ] || { export HOME=$FW/home; mkdir -p $HOME; }
S=$FW/scratch/${RESEARCH_RUN_ID:-local-$$}; rm -rf $S; mkdir -p $S || exit 1
L=$FW/cache/tmpl-unit; mkdir -p $L || exit 1
unset CARGO_TARGET_DIR
trap 'rm -rf $S' EXIT
{ lscpu | grep -E 'Model name|^CPU\(s\)'; free -g | head -2; echo "pinned=$TU_PINNED rayon=$RAYON_NUM_THREADS steer=$STEER"; } > $O/host.txt 2>&1
timed() { local name=$1; shift; if [ -x /usr/bin/time ]; then /usr/bin/time -o $O/time-$name.txt -f "TIME $name wall_s=%e maxrss_kb=%M rc=%x" "$@"; else "$@"; fi; }

LSHA=$(python3 verity/protocols/proof/flock/prover/pod/lake_closure.py key verity/protocols/proof/flock/verifier $REPO/tests/verity/protocols/proof/flock/prover/lean/TmplUnit.lean)
LEAN_BIN=$L/lean-$LSHA/verity-verify-proof; UNIT_BIN=$L/lean-$LSHA/flock-tmpl-unit; FW_BIN=$L/lean-$LSHA/flock-zk-server; LOW_BIN=$L/lean-$LSHA/flock-lower-check
(
  export PATH=$HOME/.elan/bin:$PATH
  command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
  exec 9>$L/lean.lock; flock 9
  if [ ! -x $LEAN_BIN ] || [ ! -x $UNIT_BIN ] || [ ! -x $FW_BIN ] || [ ! -x $LOW_BIN ]; then
    rm -rf $L/lean-src && LEAN_SRC=$(python3 verity/protocols/proof/flock/prover/pod/lake_closure.py copy verity/protocols/proof/flock/verifier $L/lean-src) \
      && cp tests/verity/protocols/proof/flock/prover/lean/TmplUnit.lean $LEAN_SRC/ \
      && printf '\n[[lean_exe]]\nname = "flock-tmpl-unit"\nroot = "TmplUnit"\n' >> $LEAN_SRC/lakefile.toml \
      && (cd $LEAN_SRC && lake build verity-verify-proof flock-tmpl-unit flock-zk-server flock-lower-check) && mkdir -p $L/lean-$LSHA \
      && cp $LEAN_SRC/.lake/build/bin/verity-verify-proof $LEAN_BIN && cp $LEAN_SRC/.lake/build/bin/flock-tmpl-unit $UNIT_BIN \
      && cp $LEAN_SRC/.lake/build/bin/flock-zk-server $FW_BIN && cp $LEAN_SRC/.lake/build/bin/flock-lower-check $LOW_BIN
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
export PATH=$W/py/bin:$PATH
wait $LP; [ -x $LEAN_BIN ] && [ -x $UNIT_BIN ] && [ -x $FW_BIN ] && [ -x $LOW_BIN ] || { tail -20 $O/build-lean.log; exit 1; }
( unset PYTHONPATH; timed stage $UV run -q python benchmarks/pouw/credited_served.py stage --steer "$STEER" --tn $TN --core $CORE --lower $LOW_BIN --out $S ) > $O/stage.txt 2>&1
src=$?; cat $O/time-stage.txt 2>/dev/null; grep -h '^SERVED\|^LAID\|^DRAW\|^PHASE\|^EVALUATE\|^LOWER\|^WRITER\|^READS\|^FLIPPED\|^STAGE' $O/stage.txt | cut -c1-3000
cp $S/stage.json $O/ 2>/dev/null
[ $src = 0 ] || { tail -c 4000 $O/stage.txt; exit 5; }
N=1
TREE_FROM=()
DOOR=()
for v in honest flipped; do
  D=$S/$v; Ov=$O/$v; mkdir -p $Ov
  rm -rf $CHALLENGER_WORK/challenger
  timed tmpl-$v $UNIT_BIN $D/unit.tmpl --layout iv/hybrid/4 --out $D/unit.placed > $Ov/tmpl.txt 2>&1 || { tail -40 $Ov/tmpl.txt; cat $O/time-tmpl-$v.txt; continue; }
  cat $O/time-tmpl-$v.txt; tail -3 $Ov/tmpl.txt | cut -c1-600
  PIN=$(sha512sum $D/circuit.txt | cut -d' ' -f1)
  U=(--circuit $D/circuit.txt --tmpl-unit $D/unit.placed --tables $D/tables)
  timed statement-$v $B statement --zk "${U[@]}" --instances $D/pub-$N.bin > $Ov/statement.txt 2>&1; echo "$v statement rc $?"; tail -3 $Ov/statement.txt | cut -c1-1500; cat $O/time-statement-$v.txt
  challenger_start $Ov/challenger || exit 1
  PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
  timed serve-$v $B serve --zk --listen 127.0.0.1:$PORT --out $S/sessions-$v --instances $D/pub-$N.bin "${U[@]}" --pin $PIN --operator tmpl-unit \
    --draw subset:$N --sessions 1 "${TREE_FROM[@]}" > $Ov/serve.log 2>&1 &
  SP=$!
  for i in $(seq 3600); do grep -q SERVING $Ov/serve.log && break; kill -0 $SP 2>/dev/null || break; sleep 1; done
  timed prove-$v $B prove --zk --verifier 127.0.0.1:$PORT --instances $D/inst-$N.bin "${U[@]}" --runs 1 --warm 0 > $Ov/prove.txt 2> $Ov/prove.err
  echo "$v prove rc $?"; cat $O/time-prove-$v.txt; grep -h '^LIVE' $Ov/prove.txt | cut -c1-1500; tail -5 $Ov/prove.err | cut -c1-600
  for i in $(seq 1200); do kill -0 $SP 2>/dev/null || break; sleep 1; done
  kill $SP 2>/dev/null; wait $SP; cat $O/time-serve-$v.txt; grep -h 'accepted\|VERDICT\|REJECT' $Ov/serve.log | tail -3 | cut -c1-1500
  challenger_end $Ov/challenger; echo "$v challenger exit $?"
  SESS=$(dirname $(ls -1 $S/sessions-$v/*/session.json | LC_ALL=C sort | tail -1)) || { tail -20 $Ov/serve.log; continue; }
  challenger_audit $Ov/challenger $SESS
  DOOR+=(--session $v $SESS)
  for r in registered; do
    timed lean-$v-$r $LEAN_BIN verify --zk --statement verity/flock-circuit/tmpl --circuit $D/circuit.txt --public $D/pub-$N.bin --tables $D/tables \
      --partition $D/partition.json --program $D/program.json --registered $D/$r.json --session $SESS > $Ov/lean-$r.log 2>&1
    echo "$v $r lean rc $?"; cat $O/time-lean-$v-$r.txt; grep -h '^VERDICT' $Ov/lean-$r.log | head -1 | cut -c1-1500; tail -3 $Ov/lean-$r.log | cut -c1-600
  done
  cp $D/partition.json $D/unit.tmpl $D/registered*.json $D/draw.json $Ov/; sha512sum $D/circuit.txt $D/unit.tmpl > $Ov/sha512.txt
done
( unset PYTHONPATH; timed door $UV run -q python benchmarks/pouw/credited_served.py door --lower $LOW_BIN --lean $LEAN_BIN --out $S "${DOOR[@]}" ) > $O/door.txt 2>&1
echo "door rc $?"; cat $O/time-door.txt 2>/dev/null; grep -h "^DOOR" $O/door.txt | cut -c1-2500; tail -c 3000 $O/door.txt | grep -v "^DOOR"
ls -la $S/circuit.txt; [ $(stat -c %s $S/circuit.txt 2>/dev/null || echo 0) -lt 200000000 ] && cp $S/circuit.txt $O/; true
