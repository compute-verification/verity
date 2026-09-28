#!/usr/bin/env bash
# flock class sweep: unit shapes of a served row's Q_word partition, each a verity/flock-circuit statement, proved on this pod
# (verity_flock.class_statement). Builds flock-circuit like 60-circuit.sh (GPU=1: the CUDA circuit prover), then stages and proves
# the shapes of the row's program graph (GRAPH) or of a list of Call Definitions (DEFS, JSON {descriptor id: Call count}): those in
# SHAPES (a file of shape digests), or with TABLE_FREE=1 every shape whose unit reads no pinned table, outside the token-select Calls.
# env: BINARY= PYBIN= (skip the build and the venv: a local dry run)
#      GRAPH=path/to/<row>.program.json | DEFS=path/to/defs.json  SHAPES=path/to/shapes.txt  TABLE_FREE=1  MODULES= (a module-path prefix)
#      BATCH=auto  WARM=1  RUNS=3  SELFTEST=25  JOBS=1 (shapes staged in parallel)  GPU=1  SM (from nvidia-smi)  PORT=7710
#      PROTOCOL=sweep-v0  OUT=$RESEARCH_RUN_DIR/out
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
nvidia-smi -L 2>/dev/null | tee $O/host.txt; lscpu | grep -E 'Model name|^CPU\(s\)' | tee -a $O/host.txt; free -g | tee -a $O/host.txt
G_=; [ $G = 1 ] && G_=--gpu
SRC=(--program-graph "${GRAPH:-}"); [ -n "${DEFS:-}" ] && SRC=(--definitions "$DEFS")
SEL=(); [ -n "${SHAPES:-}" ] && SEL+=(--shapes "$SHAPES"); [ "${TABLE_FREE:-0}" = 1 ] && SEL+=(--table-free)
$PYBIN -m verity_flock.class_statement "${SRC[@]}" ${MODULES:+--modules "$MODULES"} --partition q-word --out $O/classes \
  --binary $B "${SEL[@]}" --batch ${BATCH:-auto} --jobs ${JOBS:-1} --warm ${WARM:-1} --runs ${RUNS:-3} --selftest ${SELFTEST:-25} \
  --port ${PORT:-7710} --protocol ${PROTOCOL:-sweep-v0} $G_
