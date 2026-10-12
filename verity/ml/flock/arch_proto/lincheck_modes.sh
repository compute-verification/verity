#!/usr/bin/env bash
# The session verifier's lincheck modes on staged statements, on the CPU (a 0-GPU job on node 1, from the verity tree): builds
# flock-circuit (70-class-sweep.sh BUILD_ONLY=1), then runs the selftest case lincheck_modes_agree on every statement in
# FLOCK_STAGE_CACHE, smallest first, on CPUS cores of free slices (cpu-slices.sh): one honest session proved on the CPU, its
# proofs verified three times under each FC_LINCHECK mode (flat, partial, both), and two altered lincheck proofs under each.
# Writes out/lincheck_modes_agree-k<K>-n<N>.{out,err} and out/summary.jsonl (lincheck_modes.py summary).
# env: FLOCK_WORK=/workspace/jobs/proofs-arch  FLOCK_STAGE_CACHE (default FLOCK_WORK/stage-cache)  CPUS=16  SM=120
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
REPO=$(pwd); R=${RESEARCH_RUN_DIR:-/tmp/lincheck-modes}; mkdir -p $R/out
export FLOCK_WORK=${FLOCK_WORK:-/workspace/jobs/proofs-arch}; mkdir -p $FLOCK_WORK/home
export HOME=$FLOCK_WORK/home FLOCK_STAGE_CACHE=${FLOCK_STAGE_CACHE:-$FLOCK_WORK/stage-cache}
unset CARGO_TARGET_DIR
for c in $FLOCK_WORK/cuda-root /workspace/cache/flock/cuda-root; do
  [ ! -x /usr/local/cuda-13.3/bin/nvcc ] && [ -z "${CUDA_TK:-}" ] && [ -x $c/usr/local/cuda-13.3/bin/nvcc ] && export CUDA_TK=$c/usr/local/cuda-13.3
done
bins=$(OUT=$R/out/build bash verity/ml/flock/pod/70-class-sweep.sh BUILD_ONLY=1 SM=${SM:-120} 2> $R/out/build.err) || { echo "$bins"; tail -30 $R/out/build.err; exit 1; }
SB=$(echo "$bins" | sed -n 's/^SELFTEST_BINARY=//p')
sha256sum $SB | tee $R/out/binary.sha256
# the build's CUDA 13.3 runtime, as 70-class-sweep.sh runs its binaries (it links libcudart even off the GPU)
C13=${CUDA_TK:-/usr/local/cuda-13.3}; export LD_LIBRARY_PATH=$C13/compat:$C13/lib64:${LD_LIBRARY_PATH:-}
. verity/ml/flock/pod/cpu-slices.sh
hold_slices "" ${CPUS:-16}
export RAYON_NUM_THREADS=$(echo $SLICE_CPUS | tr ',' '\n' | wc -l) OMP_NUM_THREADS=$(echo $SLICE_CPUS | tr ',' '\n' | wc -l)
echo "slices=$SLICES cpus=$SLICE_CPUS wait_s=$SLICE_WAIT_S rayon=$RAYON_NUM_THREADS" | tee $R/out/host.txt
python3 $REPO/verity/ml/flock/arch_proto/lincheck_modes.py list $FLOCK_STAGE_CACHE | tee $R/out/statements.txt
ok=1
while read -r d n k; do
  o=$R/out/lincheck_modes_agree-k$k-n$n; t0=$(date +%s)
  taskset -c $SLICE_CPUS $SB selftest --instances $d/inst-$n.bin --circuit $d/circuit.txt --only lincheck_modes_agree > $o.out 2> $o.err
  rc=$?; echo "k=$k n=$n rc=$rc wall_s=$(( $(date +%s) - t0 ))"; [ $rc = 0 ] || ok=0
done < $R/out/statements.txt
python3 $REPO/verity/ml/flock/arch_proto/lincheck_modes.py summary $R/out/lincheck_modes_agree-*.out > $R/out/summary.jsonl
[ $ok = 1 ] && ! grep -q '"case_pass": false' $R/out/summary.jsonl
