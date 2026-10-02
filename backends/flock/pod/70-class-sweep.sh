#!/usr/bin/env bash
# flock class sweep: unit shapes of a served row's Q_word partition, each a verity/flock-circuit statement, proved on this pod
# (verity_flock.class_statement). Builds flock-circuit like 60-circuit.sh (GPU=1: the CUDA circuit prover), then stages and proves
# the shapes of the row's program graph (GRAPH) or of a list of Call Definitions (DEFS, JSON {descriptor id: Call count}): those in
# SHAPES (a file of shape digests), or with SELECT (table-free | reads | units) every shape whose unit reads no pinned table, reads
# one, or either, outside the token-select Calls.
# env: BINARY= PYBIN= (skip the build and the venv: a local dry run)
#      GRAPH=path/to/<row>.program.json | DEFS=path/to/defs.json | PROGRAM=path/to/module.py:FUNC (a Program's Calls)  SHAPES=path/to/shapes.txt
#      SELECT=table-free  MODULES= (a module-path prefix)
#      BATCH=auto  WARM=1  RUNS=3  SELFTEST=25  JOBS=1 (shapes staged in parallel)  GPU=1  SM (from nvidia-smi)  PORT=7710
#      SELFTEST_GPU=1 (the selftests on the GPU, its paths against each other and the CPU's included)  SELFTEST_CASES=a,b (only these)
#      SELFTEST_JOBS=1 (case processes run at once, each its own process; each parses the circuit, so memory grows with it)
#      FLOCK_WORK=/workspace (the builds, venvs and cached binaries under it)  PROTOCOL=sweep-v0  OUT=$RESEARCH_RUN_DIR/out  RESUME=path/to/results.jsonl (an earlier run's records: those shapes are skipped)
#      SELFTEST_LARGEST=0 (selftest every SELFTEST-th shape only, not also each Definition's largest)
#      BATCH_ANDS= MAX_STATEMENT_BITS= MAX_ANDS= (larger statements: --batch auto's AND budget, the witness-bit cap per statement
#      and the AND cap per shape, default 2^24)
#      N= with BATCH=fixed (instances per statement, the bit cap still applies)
#      STAGE_ONLY=1 (no GPU: build, stage the shapes into FLOCK_STAGE_CACHE and prove nothing; give SM, nvidia-smi is absent)
#      SERVE_ONLY=DIR (no GPU: stage the shapes and serve their WARM + RUNS sessions on this pod's address, published under DIR
#      for a prover in another pod that runs with FC_VERIFIERS=DIR; give SM)
#      FLOCK_PACK_WORDS=1 (narrow words share u16 leaves: four FP4 or two FP8 elements a leaf; a different statement)
#      BUILD_ONLY=1 (build, print BINARY= and SELFTEST_BINARY= and stop; give SM without a GPU)
#      ZK=1 (serve, prove and selftest with --zk)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
REPO=$(pwd); O=${OUT:-${RESEARCH_RUN_DIR:-/tmp/fcs}/out}; W=${FLOCK_WORK:-/workspace}/flock-circuit; mkdir -p $O $W
G=${GPU:-1}
SM=${SM:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d .)}; SM=${SM:-89}
if [ -z "${BINARY:-}" ]; then   # BINARY / PYBIN: a prebuilt flock-circuit and a Python with numpy (a local dry run)
  # the proving build never has seed injection; the selftest's harness gets its own build with it. Both are kept per pod, keyed
  # by the build scripts, patches and the prover's sources, so later runs of the same builds skip the compile and another
  # tree's run on the pod builds its own
  KEY=$(cat backends/flock/pod/20-gpu-link.sh backends/flock/pod/60-circuit.sh backends/flock/*.patch backends/flock/cuda_*_patch.py \
    $(find backends/flock/cuda backends/flock/live -type f | LC_ALL=C sort) | sha256sum | cut -c1-16)
  BIN=$W/bin-$KEY-g$G-sm$SM; F=${FLOCK_WORK:-/workspace}/flock-gpu-link/flock
  # the prover runs on CUDA 13.3: a driver older than that (R570) needs its compat library, cached binary or not
  C13=${CUDA_TK:-/usr/local/cuda-13.3}; export PATH=$C13/bin:$PATH NVCC=$C13/bin/nvcc LD_LIBRARY_PATH=$C13/compat:$C13/lib64:${LD_LIBRARY_PATH:-}
  # one build at a time per FLOCK_WORK (runs in parallel share its checkout); the second finds the first's binaries
  exec 8>$W/build.lock; flock 8
  # binaries this environment can't load (another machine's glibc) are rebuilt
  if [ -x $BIN/flock-circuit ] && ldd $BIN/flock-circuit $BIN/flock-circuit-selftest 2>&1 | grep -q 'not found'; then rm -rf $BIN; fi
  if [ ! -x $BIN/flock-circuit-selftest ]; then
    MODE=build GPU=$G SM=$SM bash backends/flock/pod/60-circuit.sh > $O/build.log 2>&1 || { tail -30 $O/build.log; exit 1; }
    mkdir -p $BIN && cp $F/target/release/flock-circuit $BIN/flock-circuit
    source $HOME/.cargo/env
    FEAT="--features sha512"; [ $G = 1 ] && FEAT="$FEAT --features gpu"
    (cd $F && CARGO_TARGET_DIR=$F/target-selftest cargo build --release -p flock-live $FEAT --features seed-injection --bin flock-circuit) \
      > $O/build-selftest.txt 2>&1 || { tail -30 $O/build-selftest.txt; exit 1; }
    cp $F/target-selftest/release/flock-circuit $BIN/flock-circuit-selftest
  fi
  exec 8>&-
  BINARY=$BIN/flock-circuit; SELFTEST_BINARY=$BIN/flock-circuit-selftest
fi
[ "${BUILD_ONLY:-0}" = 1 ] && { echo "BINARY=$BINARY"; echo "SELFTEST_BINARY=${SELFTEST_BINARY:-$BINARY}"; exit 0; }
B=$BINARY; SB=${SELFTEST_BINARY:-$BINARY}; sha256sum $B $SB | tee $O/binary.sha256
# ZK=1: every serve, prove and selftest under --zk (M1's masking and M2's committed coins); binary.sha256 stays the builds' own
if [ "${ZK:-0}" = 1 ]; then
  printf '#!/bin/sh\nexec %s "$@" --zk\n' "$B" > $O/prover-zk; printf '#!/bin/sh\nexec %s "$@" --zk\n' "$SB" > $O/selftest-zk
  chmod +x $O/prover-zk $O/selftest-zk; B=$O/prover-zk; SB=$O/selftest-zk
fi
# WRAP: a command the prover's prove sessions run under (71-gemm-slowdown.sh NSYS=1); binary.sha256 is the prover's own
if [ -n "${WRAP:-}" ]; then
  printf '#!/bin/sh\ncase "$1" in prove) exec %s %s "$@" ;; *) exec %s "$@" ;; esac\n' "$WRAP" "$B" "$B" > $O/prover-wrapped
  chmod +x $O/prover-wrapped; B=$O/prover-wrapped
fi
if [ -z "${PYBIN:-}" ]; then
  [ -x $HOME/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
  [ -x $W/py/bin/python ] || { $HOME/.local/bin/uv venv -q --python 3.12 $W/py && $HOME/.local/bin/uv pip install -q --python $W/py/bin/python numpy blake3; }
  PYBIN=$W/py/bin/python3
fi
export PYTHONPATH=$REPO/packages/verity/src:$REPO/backends/numerical/python:$REPO/backends/flock/python:$REPO/integrations/vllm:$REPO/tools/circuit_check/src:$REPO/protocols/sampled_proofs:$REPO/protocols/one_stage:$REPO/protocols/pouw
[ -x /usr/bin/time ] || { apt-get install -y -qq time >/dev/null 2>&1 || { apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq time >/dev/null 2>&1; }; }   # the prover's own peak memory
nvidia-smi -L 2>/dev/null | tee $O/host.txt; lscpu | grep -E 'Model name|^CPU\(s\)' | tee -a $O/host.txt; free -g | tee -a $O/host.txt
G_=; [ $G = 1 ] && G_=--gpu
# a path named at launch that isn't here is the file research run --send shipped to the run dir's inputs/
inp() { local p=$1; if [ -n "$p" ] && [ ! -e "$p" ] && [ -e "${RESEARCH_RUN_DIR:-/nonexistent}/inputs/$(basename "$p")" ]; then p=$RESEARCH_RUN_DIR/inputs/$(basename "$p"); fi; echo "$p"; }
GRAPH=$(inp "${GRAPH:-}"); DEFS=$(inp "${DEFS:-}"); SHAPES=$(inp "${SHAPES:-}"); RESUME=$(inp "${RESUME:-}")
[ -n "$RESUME" ] && { mkdir -p $O/classes; grep '^{' "$RESUME" > $O/classes/results.jsonl; }
SRC=(--program-graph "$GRAPH"); [ -n "$DEFS" ] && SRC=(--definitions "$DEFS"); [ -n "${PROGRAM:-}" ] && SRC=(--program-module "$PROGRAM")
SEL=(); [ -n "${SHAPES:-}" ] && SEL+=(--shapes "$SHAPES"); [ "${SELFTEST_LARGEST:-1}" = 0 ] && SEL+=(--no-selftest-largest)
[ -n "${BATCH_ANDS:-}" ] && SEL+=(--batch-ands "$BATCH_ANDS"); [ -n "${MAX_STATEMENT_BITS:-}" ] && SEL+=(--max-statement-bits "$MAX_STATEMENT_BITS"); [ -n "${SELECT:-}" ] && SEL+=(--select "$SELECT")
[ -n "${MAX_ANDS:-}" ] && SEL+=(--max-ands "$MAX_ANDS")
[ "${STAGE_ONLY:-0}" = 1 ] && SEL+=(--stage-only)
[ -n "${SERVE_ONLY:-}" ] && SEL+=(--serve-only "$SERVE_ONLY")
$PYBIN -m verity_flock.class_statement "${SRC[@]}" ${MODULES:+--modules "$MODULES"} --partition q-word --out $O/classes \
  --binary $B --selftest-binary $SB "${SEL[@]}" --batch ${BATCH:-auto} ${N:+--n $N} --jobs ${JOBS:-1} --warm ${WARM:-1} --runs ${RUNS:-3} --selftest ${SELFTEST:-25} \
  --port ${PORT:-7710} --protocol ${PROTOCOL:-sweep-v0} ${SELFTEST_GPU:+--selftest-gpu} ${SELFTEST_CASES:+--selftest-cases "$SELFTEST_CASES"} \
  ${SELFTEST_JOBS:+--selftest-jobs $SELFTEST_JOBS} $G_
