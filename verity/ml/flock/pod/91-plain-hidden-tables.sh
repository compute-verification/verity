#!/usr/bin/env bash
# The plain hidden-output multi-table fixture (tests/test_lean_plain_hidden_tables.py): 91-plain-hidden.sh's statement
# (tests/plain_hidden.py: RoPE d64 instances with hidden outputs, hm96-sha512 leaves, its program and partition), proved
# without --zk on OS coins in loopback sessions of two tables under a subset sample (`serve/prove --session-tables 2 --draw
# subset:K`), then Lean's `flock-verify verify --plain-tables --session-tables 2 --coins os` on them with the verifier's
# own program and partition (PROTOCOL.md §16.15, FailClosed.SoundExecHJ), and once without the flag. The staged files and
# the sessions go into the evidence store as a fixture (`research data put --kind fixture/v1 --preserve`), whose id
# `out/fixture.txt` names (else they are out/fixture). Run from the verity tree (research run --cwd source or clone); CPU
# only. Outputs under out/: host.txt, stage.json, serve.log, prove.txt, prove.err, lean-plain.log and lean-unflagged.log
# (their VERDICT lines), fixture.txt.
# Pins its own slices where it can take slice locks (node 1); on a vy-cpu pod, run it in a check slot (--pool), which pins it.
# env: CPUS=16  SESSIONS=2  K=16 (the subset sample's units, of the stage's 32: 8 a table)  FLOCK_WORK=/workspace, or where
#      /workspace isn't writable /workspace/jobs/pc-read-gadget (node 1) else /workspace/cache/plain-hidden (the prover
#      build and its venv, shared with 91-plain-hidden.sh)
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
REPO=$(pwd); O=${RESEARCH_RUN_DIR:-/tmp/plain-hidden-tables}/out; mkdir -p $O
if [ -z "${FLOCK_WORK:-}" ] && [ ! -w /workspace ]; then
  if [ -d /workspace/jobs ]; then export FLOCK_WORK=/workspace/jobs/pc-read-gadget; else export FLOCK_WORK=/workspace/cache/plain-hidden; fi
fi
FW=${FLOCK_WORK:-/workspace}; W=$FW/flock-circuit; mkdir -p $W || exit 1
[ -w "$HOME" ] || { export HOME=$FW/home; mkdir -p $HOME; }
S=$FW/scratch/${RESEARCH_RUN_ID:-local-$$}; rm -rf $S; mkdir -p $S || exit 1
L=$FW/cache/plain-hidden-tables; mkdir -p $L || exit 1
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

# the CPU prover, cached per pod under its sources' key (91-plain-hidden.sh's)
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

D=$S/stage; N=32; J=2
python3 verity/ml/flock/tests/plain_hidden.py --out $D > $O/stage.json 2> $O/stage.err || { tail -20 $O/stage.err; exit 1; }
cat $O/stage.json
PIN=$(sha512sum $D/circuit.txt | cut -d' ' -f1)
PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
$B serve --listen 127.0.0.1:$PORT --out $S/sessions --instances $D/pub-$N.bin --circuit $D/circuit.txt --tables $D/tables \
  --pin $PIN --operator plain-hidden-tables --session-tables $J --draw subset:${K:-16} --sessions ${SESSIONS:-2} > $O/serve.log 2>&1 &
SP=$!
for i in $(seq 1800); do grep -q SERVING $O/serve.log && break; kill -0 $SP 2>/dev/null || break; sleep 1; done
$B prove --verifier 127.0.0.1:$PORT --instances $D/inst-$N.bin --circuit $D/circuit.txt --tables $D/tables --session-tables $J \
  --runs ${SESSIONS:-2} --warm 0 > $O/prove.txt 2> $O/prove.err
for i in $(seq 600); do kill -0 $SP 2>/dev/null || break; sleep 1; done
kill $SP 2>/dev/null; wait $SP
mapfile -t SESS < <(dirname $(ls -1 $S/sessions/*/session.json) | LC_ALL=C sort)
[ ${#SESS[@]} -gt 0 ] || { tail -20 $O/serve.log $O/prove.err; exit 1; }

wait $LP; [ -x $LEAN_BIN ] || { tail -20 $O/build-lean.log; exit 1; }
V=(verify --statement verity/flock-circuit --coins os --circuit $D/circuit.txt --public $D/pub-$N.bin --tables $D/tables
   --session-tables $J --partition $D/partition.json --program $D/program.json)
$LEAN_BIN "${V[@]}" --plain-tables --session "${SESS[@]}" > $O/lean-plain.log 2>&1
echo "plain rc $?"; grep -h '^VERDICT' $O/lean-plain.log | cut -c1-600
$LEAN_BIN "${V[@]}" --session "${SESS[@]}" > $O/lean-unflagged.log 2>&1
echo "unflagged rc $?"; grep -h '^VERDICT' $O/lean-unflagged.log | cut -c1-600

# the fixture: the verifier's files and the sessions (no prover file: it holds the values)
X=$S/fixture; mkdir -p $X/stage $X/sessions
cp $D/circuit.txt $D/pub-$N.bin $D/program.json $D/partition.json $X/stage/ && cp -r $D/tables $X/stage/
cp -r $S/sessions/. $X/sessions/
cp $O/stage.json $X/stage.json
if [ -n "$RESEARCH" ]; then
  $RESEARCH data put --kind fixture/v1 --tree $X --preserve > $O/fixture.txt 2> $O/fixture.err || tail -5 $O/fixture.err
fi
[ -s $O/fixture.txt ] || { mkdir -p $O/fixture && cp -r $X/. $O/fixture/; echo "not in the store (no research CLI, or fixture.err): the fixture is out/fixture" > $O/fixture.txt; }
cat $O/fixture.txt
