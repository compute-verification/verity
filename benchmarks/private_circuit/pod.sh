#!/usr/bin/env bash
# Protocol 2's universal unit end to end on a CPU node (universal.py):
#   research run --on vy-nebius-1 --cwd source --project verity --declared-output 'out/*' -- \
#     bash benchmarks/private_circuit/pod.sh CLS="16,16,8 64,64,32" N=64 CPUS=128-159
# Builds C-Flock's circuit prover (CPU, 60-circuit.sh MODE=build: the proving build, `--zk` included) and the Lean verifier
# (`lake build flock-verify`) once per FLOCK_WORK, each cached under a key of its sources, then runs universal.py per class.
# env: CLS (space-separated G,N_IN,N_OUT)  N=64 (instances per class)  WARM=1 RUNS=3  SEED=0  WIRED (non-empty: universal.py --wired)
#      CPUS (a taskset CPU list the whole job runs on; nproc, the build's -j and the prover's threads follow it)
#      FLOCK_WORK=/workspace/jobs/pc-flock (builds, the Lean build directory, the venv; never the run dir)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
if [ -n "${CPUS:-}" ] && [ -z "${PC_PINNED:-}" ]; then PC_PINNED=1 exec taskset -c "$CPUS" bash "$0" "$@"; fi
set -uxo pipefail
REPO=$(pwd); O=$RESEARCH_RUN_DIR/out; W=${FLOCK_WORK:-/workspace/jobs/pc-flock}; mkdir -p $O $W/bin
nproc; free -g | head -2; lscpu | grep -E 'Model name|^CPU\(s\)' | tee $O/host.txt; echo "threads $(nproc)" | tee -a $O/host.txt

KEY=$(cat verity/ml/flock/pod/20-gpu-link.sh verity/ml/flock/pod/60-circuit.sh verity/ml/flock/*.patch verity/ml/flock/cuda_*_patch.py \
  $(find verity/ml/flock/cuda verity/ml/flock/live -type f | LC_ALL=C sort) | sha256sum | cut -c1-16)
LKEY=$(cd verity/core/service/flock && find . -path ./.lake -prune -o -type f -print | LC_ALL=C sort | xargs cat | sha256sum | cut -c1-16)
B=$W/bin/flock-circuit-$KEY; L=$W/bin/flock-verify-$LKEY
exec 8>$W/build.lock; flock 8
(
  export PATH=$HOME/.elan/bin:$PATH
  command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
  if [ ! -x $L ]; then
    rm -rf $W/lean && cp -r verity/core/service/flock $W/lean && rm -rf $W/lean/.lake
    cd $W/lean && lake build flock-verify && cp .lake/build/bin/flock-verify $L
  fi
) > $O/build-lean.log 2>&1 &
LP=$!
if [ ! -x $B ]; then
  FLOCK_WORK=$W MODE=build GPU=0 SM=0 bash verity/ml/flock/pod/60-circuit.sh > $O/build-circuit.log 2>&1 || { tail -40 $O/build-circuit.log; exit 1; }
  cp $W/flock-gpu-link/flock/target/release/flock-circuit $B
fi
wait $LP || { tail -40 $O/build-lean.log; exit 1; }
[ -x $L ] || { tail -40 $O/build-lean.log; exit 1; }
exec 8>&-
sha256sum $B $L | tee $O/binaries.sha256

[ -x $HOME/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
[ -x $W/py/bin/python ] || { $HOME/.local/bin/uv venv -q --python 3.12 $W/py && $HOME/.local/bin/uv pip install -q --python $W/py/bin/python numpy blake3; }
export RAYON_NUM_THREADS=$(nproc)
export PYTHONPATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/ml/flock/python:$REPO/integrations/vllm:$REPO/tools/verity/circuit_check/src
ARGS=(); for c in ${CLS:-16,16,8 64,64,32}; do ARGS+=(--cls "$c"); done
[ -n "${WIRED:-}" ] && ARGS+=(--wired)
$W/py/bin/python benchmarks/private_circuit/universal.py "${ARGS[@]}" --n ${N:-64} --warm ${WARM:-1} --runs ${RUNS:-3} --seed ${SEED:-0} \
  --bin $B --lean $L --out $O 2>&1 | tee $O/universal.log | cut -c1-3000
exit ${PIPESTATUS[0]}
