#!/usr/bin/env bash
# prep_job.sh starts SIZE LR | data | register: the integrator's one-time node jobs (research run --on vy-nebius-2 --cwd source).
#   starts SIZE LR   retrain the first curves' dense SIZE at LR (seed 0, data seed 0, 8 tokens per parameter), keeping theta at
#                    25%, 50% and 100% of the run in $OUT/SIZE (learning_frontier/train.py --save-theta)
#   data             prepare.py data, then SmolLM2-135M's registration
#   register SIZE RUNDIR   register SIZE50 and SIZE100 from a `starts` run's out/ (RUNDIR: the node's run directory)
set -u
P=${COVERT_PYTHON:-/workspace/jobs/venv312/bin/python}
LF=/workspace/research/scratch/learning-frontier/fineweb-edu-gpt2-v1
ROOT=${COVERT_ROOT:-/workspace/research/scratch/covert-search}
export GPU_LEASE_WHO=training:covert-search
case $1 in
  starts)
    CK=$(mktemp -d -p /workspace/research/scratch/learning-frontier ckpt-covert.XXXX)
    trap 'rm -rf "$CK"' EXIT
    bash benchmarks/learning_frontier/leased.sh "$P" --data "$LF" --ckpt-root "$CK" --out "$OUT/$2" --size "$2" --family dense \
      --lr "$3" --save-theta 0.25,0.5,1 --commit "$RESEARCH_SOURCE_SHA"
    ;;
  data)
    mkdir -p "$ROOT"
    research keep "$ROOT" --owner @training --expires never --priority keep \
      --regenerate "python benchmarks/covert_search/prepare.py data --lf-data $LF" --why "covert-search data and registered starts"
    "$P" benchmarks/covert_search/prepare.py --root "$ROOT" data --lf-data "$LF" > "$OUT/data.json" || exit $?
    gpu-lease 1 --wait --max-min 20 --mem-gb 32 -- "$P" benchmarks/covert_search/prepare.py --root "$ROOT" register --name smol \
      --hf HuggingFaceTB/SmolLM2-135M
    ;;
  register)
    for f in 050:50 100:100; do
      gpu-lease 1 --wait --max-min 20 --mem-gb 32 -- "$P" benchmarks/covert_search/prepare.py --root "$ROOT" register \
        --name "$2${f#*:}" --size "$2" --lf-theta "$3/out/$2/theta-p${f%%:*}.pt" || exit $?
    done
    ;;
esac
for d in "$ROOT"/starts/*/; do [ -f "$d/start.json" ] && cp "$d/start.json" "$OUT/start-$(basename "$d").json"; done
true
