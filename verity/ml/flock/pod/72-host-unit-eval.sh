#!/usr/bin/env bash
# The host unit evaluation (IrUnitNet::eval64_flat, and the host slots written straight from its lane groups) at the GEMM
# coordinates' largest statements: one attempt of 71-gemm-slowdown.sh with the UNITPROF profile on (FC_UNIT_PROFILE); then, in
# the same job, 71's timed sweep on the path before it (BASE, under BASE_ENV: by default FC_UNIT_FLAT=0 FC_UNIT_FUSED=0;
# out/base/result.json); one sweep per pipeline depth in DEPTHS (out/d<depth>/result.json); the sweep with every session
# building its own witness (NOPIPE: FC_PIPELINE=0, out/nopipe/result.json); then the equality check (CHECK: FC_UNIT_CHECK runs
# eval64 beside the one pass on every lane group, and packs the host slots unfused beside the fused ones, and asserts they
# agree; eval64's time in the UNITPROF lines). UNITPROF lines: out/unitprof.tsv.
# Run from the verity tree (research run --cwd source), in a Kueue job (FLOCK_WORK under /workspace/jobs).
# env: FLOCK_WORK=/workspace/jobs/flock-v2  SEED_CACHE= (a stage cache to hard-link from: statements of the same staging code)
#      TORCH= (a torch venv for the native GEMMs)  BASE=1  BASE_ENV=  DEPTHS=  NOPIPE=1  CHECK=1  and any KEY=VALUE for
#      71-gemm-slowdown.sh
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
R=${RESEARCH_RUN_DIR:-/tmp/hue}; mkdir -p $R/out
export FLOCK_WORK=${FLOCK_WORK:-/workspace/jobs/flock-v2}; mkdir -p $FLOCK_WORK/home $FLOCK_WORK/flock-circuit
export HOME=$FLOCK_WORK/home FLOCK_STAGE_CACHE=${FLOCK_STAGE_CACHE:-$FLOCK_WORK/stage-cache}; mkdir -p $FLOCK_STAGE_CACHE
[ -n "${SEED_CACHE:-}" ] && cp -al -n $SEED_CACHE/. $FLOCK_STAGE_CACHE/
[ -n "${TORCH:-}" ] && [ ! -e $FLOCK_WORK/flock-circuit/torch ] && ln -s $TORCH $FLOCK_WORK/flock-circuit/torch
FC_UNIT_PROFILE=1 bash verity/ml/flock/pod/71-gemm-slowdown.sh "$@" || exit 1
# 71's toolkit and threads for the sweeps after it (its binaries are cached under FLOCK_WORK by then)
for c in $FLOCK_WORK/cuda-root /workspace/cache/flock/cuda-root; do
  [ -z "${CUDA_TK:-}" ] && [ ! -x /usr/local/cuda-13.3/bin/nvcc ] && [ -x $c/usr/local/cuda-13.3/bin/nvcc ] && export CUDA_TK=$c/usr/local/cuda-13.3
done
unset CARGO_TARGET_DIR
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-${OMP_NUM_THREADS:-$(nproc)}}
SW=(DEFS=${DEFS:-backends/flock/pod/gemm-coordinates.defs.json} BATCH=auto BATCH_ANDS=${BATCH_ANDS:-4294967296}
    MAX_STATEMENT_BITS=${MAX_STATEMENT_BITS:-17179869184} SELFTEST=0)
PY=$FLOCK_WORK/flock-circuit/py/bin/python3
# each later sweep's class_statement writes its result.json to its own RESEARCH_RUN_DIR: the attempt's result.json stays 71's
# sweep NAME's slowdowns: out/NAME/result.json, and out/NAME/out/slowdown.json (compute writes it to out/ beside its --result)
slowdown() {
  mkdir -p $R/out/$1/out
  $PY verity/ml/flock/pod/gemm_slowdown.py compute --sweep $R/out/$1/classes/results.jsonl --native $R/out/native.json \
    --result $R/out/$1/result.json
}
if [ "${BASE:-1}" = 1 ]; then
  env ${BASE_ENV:-FC_UNIT_FLAT=0 FC_UNIT_FUSED=0} FC_UNIT_PROFILE=1 OUT=$R/out/base RESEARCH_RUN_DIR=$R/out/base bash verity/ml/flock/pod/70-class-sweep.sh "${SW[@]}" \
    WARM=${WARM:-1} RUNS=${RUNS:-3} || exit 1
  slowdown base || exit 1
fi
for d in ${DEPTHS:-}; do
  FC_UNIT_PROFILE=1 FC_PIPELINE_DEPTH=$d OUT=$R/out/d$d RESEARCH_RUN_DIR=$R/out/d$d bash verity/ml/flock/pod/70-class-sweep.sh "${SW[@]}" \
    WARM=${WARM:-1} RUNS=${RUNS:-3} || exit 1
  slowdown d$d || exit 1
done
if [ "${NOPIPE:-1}" = 1 ]; then
  FC_UNIT_PROFILE=1 FC_PIPELINE=0 OUT=$R/out/nopipe RESEARCH_RUN_DIR=$R/out/nopipe bash verity/ml/flock/pod/70-class-sweep.sh "${SW[@]}" WARM=${WARM:-1} RUNS=${RUNS:-3} || exit 1
  slowdown nopipe || exit 1
fi
if [ "${CHECK:-1}" = 1 ]; then
  FC_UNIT_CHECK=1 FC_UNIT_PROFILE=1 FC_PIPELINE=0 OUT=$R/out/check RESEARCH_RUN_DIR=$R/out/check bash verity/ml/flock/pod/70-class-sweep.sh "${SW[@]}" WARM=0 RUNS=1 || exit 1
  $PY -c "import json, sys; rs = [json.loads(l) for l in open(sys.argv[1]) if l.startswith('{')]
sys.exit(0 if rs and all((r.get('live') or {}).get('accepted') for r in rs) else 1)" $R/out/check/classes/results.jsonl \
    || { echo "a statement of the check sweep was not accepted"; exit 1; }
fi
grep -H UNITPROF $R/out/*/classes/*/prove.err > $R/out/unitprof.tsv
