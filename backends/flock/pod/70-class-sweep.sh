#!/usr/bin/env bash
# flock class sweep: unit shapes of a served row's Q_word partition, each a verity/flock-circuit statement, proved on this pod
# (verity_flock.class_statement). Builds flock-circuit like 60-circuit.sh (GPU=1: the CUDA circuit prover), then stages and proves
# the shapes of the row's program graph (GRAPH) or of a list of Call Definitions (DEFS, JSON {descriptor id: Call count}): those in
# SHAPES (a file of shape digests), or with SELECT (table-free | reads | units) every shape whose unit reads no pinned table, reads
# one, or either, outside the token-select Calls.
# env: BINARY= PYBIN= (skip the build and the venv: a local dry run)
#      GRAPH=path/to/<row>.program.json | DEFS=path/to/defs.json  SHAPES=path/to/shapes.txt  SELECT=table-free  MODULES= (a module-path prefix)
#      BATCH=auto  WARM=1  RUNS=3  SELFTEST=25  JOBS=1 (shapes staged in parallel)  GPU=1  SM (from nvidia-smi)  PORT=7710
#      PROTOCOL=sweep-v0  OUT=$RESEARCH_RUN_DIR/out
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
REPO=$(pwd); O=${OUT:-${RESEARCH_RUN_DIR:-/tmp/fcs}/out}; W=/workspace/flock-circuit; mkdir -p $O $W
G=${GPU:-1}
SM=${SM:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d .)}; SM=${SM:-89}
if [ -z "${BINARY:-}" ]; then   # BINARY / PYBIN: a prebuilt flock-circuit and a Python with numpy (a local dry run)
  # the proving build never has seed injection; the selftest's harness gets its own build with it. Both are kept per pod, keyed
  # by the build scripts and patches, so later runs of the same commit's builds skip the compile
  KEY=$(cat backends/flock/pod/20-gpu-link.sh backends/flock/pod/60-circuit.sh backends/flock/*.patch backends/flock/cuda_*_patch.py | sha256sum | cut -c1-16)
  BIN=$W/bin-$KEY-g$G-sm$SM; F=/workspace/flock-gpu-link/flock
  if [ ! -x $BIN/flock-circuit-selftest ]; then
    MODE=build GPU=$G SM=$SM bash backends/flock/pod/60-circuit.sh > $O/build.log 2>&1 || { tail -30 $O/build.log; exit 1; }
    mkdir -p $BIN && cp $F/target/release/flock-circuit $BIN/flock-circuit
    source $HOME/.cargo/env; C13=${CUDA_TK:-/usr/local/cuda-13.3}; export PATH=$C13/bin:$PATH NVCC=$C13/bin/nvcc LD_LIBRARY_PATH=$C13/compat:$C13/lib64:${LD_LIBRARY_PATH:-}
    FEAT="--features sha512"; [ $G = 1 ] && FEAT="$FEAT --features gpu"
    (cd $F && CARGO_TARGET_DIR=$F/target-selftest cargo build --release -p flock-live $FEAT --features seed-injection --bin flock-circuit) \
      > $O/build-selftest.txt 2>&1 || { tail -30 $O/build-selftest.txt; exit 1; }
    cp $F/target-selftest/release/flock-circuit $BIN/flock-circuit-selftest
  fi
  BINARY=$BIN/flock-circuit; SELFTEST_BINARY=$BIN/flock-circuit-selftest
fi
B=$BINARY; SB=${SELFTEST_BINARY:-$BINARY}; sha256sum $B $SB | tee $O/binary.sha256
if [ -z "${PYBIN:-}" ]; then
  [ -x $HOME/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
  [ -x $W/py/bin/python ] || { $HOME/.local/bin/uv venv -q --python 3.12 $W/py && $HOME/.local/bin/uv pip install -q --python $W/py/bin/python numpy blake3; }
  PYBIN=$W/py/bin/python3
fi
export PYTHONPATH=$REPO/packages/verity/src:$REPO/backends/numerical/python:$REPO/backends/flock/python:$REPO/integrations/vllm:$REPO/tools/circuit_check/src:$REPO/protocols/sampled_proofs:$REPO/protocols/one_stage
[ -x /usr/bin/time ] || { apt-get install -y -qq time >/dev/null 2>&1 || { apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq time >/dev/null 2>&1; }; }   # the prover's own peak memory
nvidia-smi -L 2>/dev/null | tee $O/host.txt; lscpu | grep -E 'Model name|^CPU\(s\)' | tee -a $O/host.txt; free -g | tee -a $O/host.txt
G_=; [ $G = 1 ] && G_=--gpu
# a path named at launch that isn't here is the file research run --send shipped to the run dir's inputs/
inp() { local p=$1; if [ -n "$p" ] && [ ! -e "$p" ] && [ -e "${RESEARCH_RUN_DIR:-/nonexistent}/inputs/$(basename "$p")" ]; then p=$RESEARCH_RUN_DIR/inputs/$(basename "$p"); fi; echo "$p"; }
GRAPH=$(inp "${GRAPH:-}"); DEFS=$(inp "${DEFS:-}"); SHAPES=$(inp "${SHAPES:-}")
SRC=(--program-graph "$GRAPH"); [ -n "$DEFS" ] && SRC=(--definitions "$DEFS")
SEL=(); [ -n "${SHAPES:-}" ] && SEL+=(--shapes "$SHAPES"); [ -n "${SELECT:-}" ] && SEL+=(--select "$SELECT")
$PYBIN -m verity_flock.class_statement "${SRC[@]}" ${MODULES:+--modules "$MODULES"} --partition q-word --out $O/classes \
  --binary $B --selftest-binary $SB "${SEL[@]}" --batch ${BATCH:-auto} --jobs ${JOBS:-1} --warm ${WARM:-1} --runs ${RUNS:-3} --selftest ${SELFTEST:-25} \
  --port ${PORT:-7710} --protocol ${PROTOCOL:-sweep-v0} $G_
