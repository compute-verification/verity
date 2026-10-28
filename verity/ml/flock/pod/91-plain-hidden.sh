#!/usr/bin/env bash
# The plain hidden-output fixture (tests/plain_hidden.py, tests/test_lean_plain_hidden.py): a RoPE d64 statement with hidden
# outputs, no shared rows and no registered value, proved in one loopback session without --zk under a subset sample on OS
# coins, then Lean's `flock-verify verify --coins os` on it with the verifier's own program and partition (PROTOCOL.md
# §16.15: the plain claim at hidden outputs draws by the subset law at one table, FailClosed.SoundExecH at its subset sample design). The staged files
# and the session go into the evidence store as a fixture (`research data put --kind fixture/v1 --preserve`), whose id
# `out/fixture.txt` names. Run from the verity tree (research run --cwd source); CPU only. Outputs under out/: host.txt,
# stage.json, statement.txt, serve.log, prove.txt and prove.err, live.json (the LIVE record), lean-honest.log (its VERDICT
# line), fixture.txt.
# Pins its own slices where it can take slice locks (node 1); on a vy-cpu pod, run it in a check slot (--pool), which pins it.
# env: CPUS=16  K=16 (the subset sample's units, of the stage's 32)  FLOCK_WORK=/workspace, or where /workspace isn't writable
#      /workspace/jobs/pc-read-gadget (node 1, shared as 76-read-gadget.sh's) else /workspace/cache/plain-hidden (the prover
#      build and its venv)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
if [ -z "${PH_PINNED:-}" ]; then
  if mkdir -p "${SLICE_LOCKS:-/workspace/jobs/slices}" 2>/dev/null; then
    . verity/ml/flock/pod/cpu-slices.sh
    hold_slices "" "${CPUS:-16}" || exit 1
    export PH_PINNED=$SLICE_CPUS RAYON_NUM_THREADS=$(echo $SLICE_CPUS | tr ',' '\n' | wc -l)
    exec taskset -c $SLICE_CPUS bash "$0" "$@"
  fi
  # no slice locks here (a vy-cpu pod): the check slot (research run --pool) has pinned the job to its own cores
  export PH_PINNED=slot RAYON_NUM_THREADS=$(nproc)
fi
export FC_COINS=os
REPO=$(pwd); O=${RESEARCH_RUN_DIR:-/tmp/plain-hidden}/out; mkdir -p $O
if [ -z "${FLOCK_WORK:-}" ] && [ ! -w /workspace ]; then
  if [ -d /workspace/jobs ]; then export FLOCK_WORK=/workspace/jobs/pc-read-gadget; else export FLOCK_WORK=/workspace/cache/plain-hidden; fi
fi
FW=${FLOCK_WORK:-/workspace}; W=$FW/flock-circuit; mkdir -p $W || exit 1
[ -w "$HOME" ] || { export HOME=$FW/home; mkdir -p $HOME; }
S=$FW/scratch/${RESEARCH_RUN_ID:-local-$$}; rm -rf $S; mkdir -p $S || exit 1
L=$FW/cache/plain-hidden; mkdir -p $L || exit 1
unset CARGO_TARGET_DIR
trap 'rm -rf $S' EXIT
{ lscpu | grep -E 'Model name|^CPU\(s\)'; echo "pinned=$PH_PINNED rayon=$RAYON_NUM_THREADS slices=${SLICES:-}"; } > $O/host.txt 2>&1

# Lean's flock-verify, built once per source, while the prover builds
LSHA=$(cd verity/core/service/flock && find . -name .lake -prune -o -type f -print | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)
LEAN_BIN=$L/lean-$LSHA/flock-verify
(
  export PATH=$HOME/.elan/bin:$PATH
  command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
  exec 9>$L/lean.lock; flock 9
  if [ ! -x $LEAN_BIN ]; then
    rm -rf $L/lean-src && mkdir -p $L/lean-src && cp -r verity/core/service/flock/. $L/lean-src/ \
      && (cd $L/lean-src && lake build flock-verify) && mkdir -p $L/lean-$LSHA && cp $L/lean-src/.lake/build/bin/flock-verify $LEAN_BIN
  fi
) > $O/build-lean.log 2>&1 &
LP=$!

# the CPU prover, cached per pod under its sources' key (76-read-gadget.sh's)
SM=${SM:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d .)}; SM=${SM:-89}
KEY=$(cat verity/ml/flock/pod/20-gpu-link.sh verity/ml/flock/pod/60-circuit.sh verity/ml/flock/*.patch verity/ml/flock/cuda_*_patch.py \
  $(find verity/ml/flock/cuda verity/ml/flock/live -type f | LC_ALL=C sort) | sha256sum | cut -c1-16)
BIN=$W/bin-$KEY-g0-sm$SM; F=$FW/flock-gpu-link/flock
exec 8>$W/build.lock; flock 8
if [ -x $BIN/flock-circuit ] && [[ $(ldd $BIN/flock-circuit 2>&1) == *"not found"* ]]; then
  rm -rf $BIN || { echo "$BIN: built against another glibc, and this job can't remove it" >&2; exit 1; }
fi
if [ ! -x $BIN/flock-circuit ]; then
  RESEARCH_RUN_DIR=$O/build FLOCK_WORK=$FW MODE=build GPU=0 SM=$SM bash verity/ml/flock/pod/60-circuit.sh > $O/build.log 2>&1 || { tail -30 $O/build.log; exit 1; }
  mkdir -p $BIN && cp $F/target/release/flock-circuit $BIN/flock-circuit
fi
exec 8>&-
B=$BIN/flock-circuit; sha256sum $B | tee $O/binary.sha256
UV=$(command -v uv || echo $HOME/.local/bin/uv)
[ -x $UV ] || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; UV=$HOME/.local/bin/uv; }
[ -x $W/py/bin/python ] || { $UV venv -q --python 3.12 $W/py && $UV pip install -q --python $W/py/bin/python numpy blake3; } || exit 1
RESEARCH=$(command -v research || true)
export PATH=$W/py/bin:$PATH PYTHONPATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/ml/flock/python:$REPO/integrations/vllm

D=$S/stage; N=32
python3 verity/ml/flock/tests/plain_hidden.py --out $D > $O/stage.json 2> $O/stage.err || { tail -20 $O/stage.err; exit 1; }
PIN=$(sha512sum $D/circuit.txt | cut -d' ' -f1)
$B statement --circuit $D/circuit.txt --instances $D/pub-$N.bin --tables $D/tables > $O/statement.txt 2>&1
PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
$B serve --listen 127.0.0.1:$PORT --out $S/sessions --instances $D/pub-$N.bin --circuit $D/circuit.txt --tables $D/tables \
  --pin $PIN --operator plain-hidden --draw subset:${K:-16} --sessions 1 > $O/serve.log 2>&1 &
SP=$!
for i in $(seq 1800); do grep -q SERVING $O/serve.log && break; kill -0 $SP 2>/dev/null || break; sleep 1; done
$B prove --verifier 127.0.0.1:$PORT --instances $D/inst-$N.bin --circuit $D/circuit.txt --tables $D/tables --runs 1 --warm 0 \
  > $O/prove.txt 2> $O/prove.err
grep -h '^LIVE' $O/prove.txt | head -1 | cut -f2- > $O/live.json
for i in $(seq 600); do kill -0 $SP 2>/dev/null || break; sleep 1; done
kill $SP 2>/dev/null; wait $SP
SESS=$(dirname $(ls -1 $S/sessions/*/session.json | LC_ALL=C sort | tail -1)) || exit 1

wait $LP; [ -x $LEAN_BIN ] || { tail -20 $O/build-lean.log; exit 1; }
$LEAN_BIN verify --statement verity/flock-circuit --coins os --circuit $D/circuit.txt --public $D/pub-$N.bin \
  --partition $D/partition.json --program $D/program.json --session $SESS > $O/lean-honest.log 2>&1
echo "honest rc $?"
grep -h '^VERDICT' $O/lean-honest.log | head -1 | cut -c1-600

# the fixture: the verifier's files and the session (no prover file: it holds the values)
X=$S/fixture; mkdir -p $X/stage $X/session
cp $D/circuit.txt $D/pub-$N.bin $D/program.json $D/partition.json $X/stage/ && cp -r $D/tables $X/stage/
cp -r $SESS/. $X/session/
cp $O/live.json $X/live.json
if [ -n "$RESEARCH" ]; then
  $RESEARCH data put --kind fixture/v1 --tree $X --preserve > $O/fixture.txt 2> $O/fixture.err || tail -5 $O/fixture.err
fi
[ -s $O/fixture.txt ] || { mkdir -p $O/fixture && cp -r $X/. $O/fixture/; echo "not in the store (no research CLI, or fixture.err): the fixture is out/fixture" > $O/fixture.txt; }
cat $O/fixture.txt
