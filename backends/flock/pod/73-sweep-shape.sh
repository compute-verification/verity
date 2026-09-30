#!/usr/bin/env bash
# The backend sweep as one Kueue job per unit shape (vy-nebius-1's dispatcher), over a served row's Build.
#   MODE=cut   ROW=<row slug> BUILD=<the row's Build dir, with build_request/instances.json.gz>
#              the row's program graph (verity_vllm program-graph) and class_statement --counts-only under SELECT, into
#              $SWEEP/$ROW/ (program.json, counts.json, shapes.tsv: shape, word gates, row units, its Definitions) and the run's out/
#   MODE=stage ROW=<row slug> SHAPE=<shape digest>   (no GPU)
#              builds the prover and stages that one shape into FLOCK_STAGE_CACHE, from the Definitions of $SWEEP/$ROW/counts.json
#              that hold it, at their row Call counts
#   MODE=shape ROW=<row slug> SHAPE=<shape digest>   (one GPU)
#              proves it (70-class-sweep.sh) with the same settings, so a staged shape comes from the cache; SELFTEST=1
#              SELFTEST_GPU=1 for its selftest on the GPU (the device's proofs and transcripts against the CPU's, byte for byte)
# env: SWEEP=/workspace/jobs/sweep2  SELECT=units  FLOCK_WORK=/workspace/jobs/flock-sweep2  SM=120  and 70-class-sweep.sh's (SELFTEST=0 here)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
REPO=$(pwd); R=${RESEARCH_RUN_DIR:-/tmp/sweep-shape}; mkdir -p $R/out
SWEEP=${SWEEP:-/workspace/jobs/sweep2}; D=$SWEEP/$ROW; mkdir -p $D
export FLOCK_WORK=${FLOCK_WORK:-/workspace/jobs/flock-sweep2}; mkdir -p $FLOCK_WORK/home; export HOME=$FLOCK_WORK/home
export FLOCK_STAGE_CACHE=${FLOCK_STAGE_CACHE:-$FLOCK_WORK/stage-cache} SM=${SM:-120}
# the prover as flock-m0-v3 measures it (#554, attempt #15): 4x4 GEMM tiles and m = 35 statements, witnesses built 4 ahead
export FLOCK_GEMM_TILE=${FLOCK_GEMM_TILE:-4x4} BATCH_ANDS=${BATCH_ANDS:-34359738368} MAX_STATEMENT_BITS=${MAX_STATEMENT_BITS:-34359738368} \
  FC_PIPELINE_DEPTH=${FC_PIPELINE_DEPTH:-4} FC_HOST_PREPIN=${FC_HOST_PREPIN:-1}
unset CARGO_TARGET_DIR
[ -n "${RAYON_NUM_THREADS:-}" ] || export RAYON_NUM_THREADS=${OMP_NUM_THREADS:-$(nproc)}
for c in $FLOCK_WORK/cuda-root /workspace/cache/flock/cuda-root; do
  [ ! -x /usr/local/cuda-13.3/bin/nvcc ] && [ -z "${CUDA_TK:-}" ] && [ -x $c/usr/local/cuda-13.3/bin/nvcc ] && export CUDA_TK=$c/usr/local/cuda-13.3
done
{ nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv; nproc; } | tee $R/out/host.txt
export PYTHONPATH=$REPO/packages/verity/src:$REPO/backends/numerical/python:$REPO/backends/flock/python:$REPO/integrations/vllm:$REPO/tools/circuit_check/src:$REPO/protocols/sampled_proofs:$REPO/protocols/one_stage:$REPO/protocols/pouw
case ${MODE:?cut or shape} in
cut)
  W=$FLOCK_WORK/flock-circuit; mkdir -p $W; UV=$(command -v uv || echo /workspace/jobs/bin/uv)
  [ -x $W/py/bin/python ] || { $UV venv -q --python 3.12 $W/py && $UV pip install -q --python $W/py/bin/python numpy blake3; } || exit 1
  PY=$W/py/bin/python3
  $PY -m verity_vllm.pipeline.cli program-graph $R/out/program.json $BUILD/build_request/instances.json.gz \
    --row "{\"row\": \"$ROW\"}" || exit 1
  echo "graph_s=$SECONDS" | tee $R/out/cut-timing.txt
  $PY -m verity_flock.class_statement --program-graph $R/out/program.json --partition q-word --out $R/out/cut \
    --select ${SELECT:-units} --counts-only || exit 1
  echo "cut_s=$SECONDS" | tee -a $R/out/cut-timing.txt
  $PY - $R/out/cut/counts.json $R/out/shapes.tsv <<'EOF' || exit 1
import json, sys
c = json.load(open(sys.argv[1]))
defs = {}
for fid, per in c["per_definition"].items():
    for sh in per:
        defs.setdefault(sh, []).append(fid)
rows = sorted(c["shapes"].items(), key=lambda kv: (kv[1]["word_gates"], kv[0]))
with open(sys.argv[2], "w") as f:
    for sh, s in rows:
        f.write(f"{sh}\t{s['word_gates']}\t{s['row_units']}\t{json.dumps(sorted(defs.get(sh, [])))}\n")
print(f"{len(rows)} shapes, {sum(s['row_units'] or 0 for _, s in rows)} row units")
EOF
  cp $R/out/program.json $R/out/cut/counts.json $R/out/shapes.tsv $D/ ;;
stage|shape)
  $(command -v python3) - $D/counts.json $SHAPE $R/out/defs.json <<'EOF' || exit 1
import json, sys
c, s = json.load(open(sys.argv[1])), sys.argv[2]
holders = {fid: per for fid, per in c["per_definition"].items() if s in per}
if not holders:
    sys.exit(f"shape {s} is in no Definition of {sys.argv[1]}")
# the smallest Definition holding it (attention's shared shapes are in up to 1,151 of them), its Calls scaled to the shape's row
# units: those cap the batch; the row's own count stays counts.json's, which the rollup reads
fid = min(holders, key=lambda f: (sum(holders[f].values()), f))
row = c["shapes"][s]["row_units"] or holders[fid][s] * c["calls"][fid]
json.dump({fid: max(1, round(row / holders[fid][s]))}, open(sys.argv[3], "w"))
EOF
  echo $SHAPE > $R/out/shapes.txt
  OUT=$R/out bash backends/flock/pod/70-class-sweep.sh DEFS=$R/out/defs.json SHAPES=$R/out/shapes.txt SELECT=${SELECT:-units} \
    SELFTEST=${SELFTEST:-0} SELFTEST_LARGEST=0 STAGE_ONLY=$([ $MODE = stage ] && echo 1 || echo 0) || exit 1 ;;
esac
