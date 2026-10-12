#!/usr/bin/env bash
# A serve session on statements already staged, as they are (a 1-GPU job on node 1, from the verity tree): builds or finds
# flock-circuit (70-class-sweep.sh BUILD_ONLY=1), then class_statement.prove on each directory in STMTS on CPUS cores of free
# slices (cpu-slices.sh), WARM warm-up and RUNS timed sessions each, as 70-class-sweep.sh runs them (serve_staged.py). For
# statements another tree staged, such as M0 #20's (the flock-m0 stage cache): a stage-cache key hashes the staging code, so
# a current tree's 70-class-sweep.sh would stage them again. Writes out/<shape16>/ and out/summary.jsonl (serve_staged.py).
# env: STMTS="dir dir" (required)  FLOCK_WORK=/workspace/jobs/proofs-arch  CPUS=16  SM=120  WARM=1  RUNS=3  PORT=7710
#      DISK_MAX=80 (don't start with /workspace this full, in %); any other KEY=VALUE reaches the prover's and the verifier's
#      environment (FLOCK_GEMM_TILE, FC_PIPELINE_DEPTH, FC_HOST_PREPIN, FC_LINCHECK, ...)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
: "${STMTS:?STMTS= is required}"
REPO=$(pwd); R=${RESEARCH_RUN_DIR:-/tmp/serve-staged}; mkdir -p $R/out
use=$(df --output=pcent /workspace 2>/dev/null | tail -1 | tr -dc 0-9)
if [ -n "$use" ] && [ "$use" -ge "${DISK_MAX:-80}" ]; then echo "/workspace is at $use%: not starting"; exit 1; fi
export FLOCK_WORK=${FLOCK_WORK:-/workspace/jobs/proofs-arch}; mkdir -p $FLOCK_WORK/home
export HOME=$FLOCK_WORK/home
unset CARGO_TARGET_DIR
for c in $FLOCK_WORK/cuda-root /workspace/cache/flock/cuda-root; do
  [ ! -x /usr/local/cuda-13.3/bin/nvcc ] && [ -z "${CUDA_TK:-}" ] && [ -x $c/usr/local/cuda-13.3/bin/nvcc ] && export CUDA_TK=$c/usr/local/cuda-13.3
done
bins=$(OUT=$R/out/build bash verity/ml/flock/pod/70-class-sweep.sh BUILD_ONLY=1 SM=${SM:-120} 2> $R/out/build.err) || { echo "$bins"; tail -30 $R/out/build.err; exit 1; }
B=$(echo "$bins" | sed -n 's/^BINARY=//p')
sha256sum $B | tee $R/out/binary.sha256
C13=${CUDA_TK:-/usr/local/cuda-13.3}; export LD_LIBRARY_PATH=$C13/compat:$C13/lib64:${LD_LIBRARY_PATH:-}
. verity/ml/flock/pod/cpu-slices.sh
hold_slices "" ${CPUS:-16}
export RAYON_NUM_THREADS=$(echo $SLICE_CPUS | tr ',' '\n' | wc -l) OMP_NUM_THREADS=$(echo $SLICE_CPUS | tr ',' '\n' | wc -l)
{ nvidia-smi -L; echo "cuda_visible=${CUDA_VISIBLE_DEVICES:-} slices=$SLICES cpus=$SLICE_CPUS wait_s=$SLICE_WAIT_S rayon=$RAYON_NUM_THREADS"
  env | grep -E '^(FC_|FLOCK_GEMM_TILE=)' | sort; } | tee $R/out/host.txt
export PYTHONPATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/ml/flock/python:$REPO/integrations/vllm:$REPO/tools/verity/circuit_check/src:$REPO/catalog:$REPO/kernels:$REPO/experimental:$REPO/protocols/one_stage
S=$FLOCK_WORK/scratch/${RESEARCH_RUN_ID:-serve-staged-$$}; rm -rf $S; mkdir -p $S
trap 'rm -rf $S' EXIT
taskset -c $SLICE_CPUS $FLOCK_WORK/flock-circuit/py/bin/python3 $REPO/verity/ml/flock/arch_proto/serve_staged.py $B $S $R/out ${PORT:-7710} \
  ${WARM:-1} ${RUNS:-3} $STMTS
