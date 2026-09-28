#!/usr/bin/env bash
# flock class sweep: every listed unit shape of a served row's Q_word partition as a verity/flock-circuit statement, proved on this
# pod (verity_flock.class_statement). Builds flock-circuit like 60-circuit.sh (GPU=1: the CUDA circuit prover), then stages and proves
# each shape in SHAPES (a file of shape digests; the row's program graph from GRAPH, sent with the run or in the shipped tree).
# env: BINARY= PYBIN= (skip the build and the venv: a local dry run)
#      GRAPH=path/to/<row>.program.json  SHAPES=path/to/shapes.txt  MODULES= (a module-path prefix)  BATCH=auto  WARM=1  RUNS=3
#      SELFTEST=25  JOBS=1 (shapes staged in parallel)  GPU=1  SM (from nvidia-smi)  PORT=7710  OUT=$RESEARCH_RUN_DIR/out
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
REPO=$(pwd); O=${OUT:-${RESEARCH_RUN_DIR:-/tmp/fcs}/out}; W=/workspace/flock-circuit; mkdir -p $O $W
G=${GPU:-1}
SM=${SM:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d .)}; SM=${SM:-89}
if [ -z "${BINARY:-}" ]; then   # BINARY / PYBIN: a prebuilt flock-circuit and a Python with numpy (a local dry run)
  MODE=build GPU=$G SM=$SM bash backends/flock/pod/60-circuit.sh > $O/build.log 2>&1 || { tail -30 $O/build.log; exit 1; }
  BINARY=/workspace/flock-gpu-link/flock/target/release/flock-circuit
fi
B=$BINARY; sha256sum $B | tee $O/binary.sha256
if [ -z "${PYBIN:-}" ]; then
  [ -x $W/py/bin/python ] || { $HOME/.local/bin/uv venv -q --python 3.12 $W/py && $HOME/.local/bin/uv pip install -q --python $W/py/bin/python numpy blake3; }
  PYBIN=$W/py/bin/python3
fi
export PYTHONPATH=$REPO/packages/verity/src:$REPO/backends/numerical/python:$REPO/backends/flock/python:$REPO/integrations/vllm:$REPO/tools/circuit_check/src
nvidia-smi -L 2>/dev/null | tee $O/host.txt; lscpu | grep -E 'Model name|^CPU\(s\)' | tee -a $O/host.txt
G_=; [ $G = 1 ] && G_=--gpu
$PYBIN -m verity_flock.class_statement --program-graph "$GRAPH" ${MODULES:+--modules "$MODULES"} --partition q-word --out $O/classes \
  --binary $B --shapes "$SHAPES" --batch ${BATCH:-auto} --jobs ${JOBS:-1} --warm ${WARM:-1} --runs ${RUNS:-3} --selftest ${SELFTEST:-25} --port ${PORT:-7710} $G_
