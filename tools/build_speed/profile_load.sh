#!/usr/bin/env bash
# profile_load.py over a kept row dir (BENCH_KEEP=NAME of bench_build.sh): $BASE/keep/NAME/ROW. Measurement only.
set -euo pipefail
NAME=$1 ROW=$2
SRC=$(pwd)
export PY=/workspace/jobs/venv312/bin/python OMP_NUM_THREADS=16
export PYTHONPATH=$SRC/integrations/vllm:$SRC/packages/verity/src:$SRC/tools/research/src
mkdir -p "$RESEARCH_RUN_DIR/out"
"$PY" "$SRC/tools/build_speed/profile_load.py" "/workspace/research/build-speed/keep/$NAME/$ROW" "$RESEARCH_RUN_DIR/out"
