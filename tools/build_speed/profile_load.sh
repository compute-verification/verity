#!/usr/bin/env bash
# profile_load.py over a kept row dir (BENCH_KEEP=NAME of bench_build.sh): $BASE/keep/NAME/ROW, then the whole workload compose
# (`global-program`, no --follow) under cProfile into scratch. Measurement only.
set -euo pipefail
NAME=$1 ROW=$2
SRC=$(pwd)
RID=${RESEARCH_RUN_ID:?}
SCR=/workspace/research/build-speed/runs/$RID
mkdir -p "$SCR" "$RESEARCH_RUN_DIR/out"
trap 'rm -rf "$SCR"' EXIT
export PY=/workspace/jobs/venv312/bin/python OMP_NUM_THREADS=16 TMPDIR=$SCR
export PYTHONPATH=$SRC/integrations/vllm:$SRC/packages/verity/src:$SRC/tools/research/src
D=/workspace/research/build-speed/keep/$NAME/$ROW
"$PY" "$SRC/tools/build_speed/profile_load.py" "$D" "$RESEARCH_RUN_DIR/out"
cd "$SRC/integrations/vllm"
args=()
for d in "$D"/build_request_LP*_T*; do [ -d "$d" ] && args+=(--derived "$d"); done
t0=$(date +%s.%N)
"$PY" -m cProfile -o "$RESEARCH_RUN_DIR/out/compose.prof" -m verity_vllm.pipeline.cli global-program --workload "workloads/$ROW.json" "${args[@]}" \
  --out "$SCR/build_workload" --checkpoint-dir /workspace/hf/hub/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da \
  > "$RESEARCH_RUN_DIR/out/compose.log" 2>&1
echo "compose_wall_s=$(python3 -c "import sys,time; print(round(time.time()-float(sys.argv[1]),1))" "$t0")" | tee "$RESEARCH_RUN_DIR/out/compose_wall.txt"
"$PY" -c "import pstats,sys; s=pstats.Stats(sys.argv[1]); s.sort_stats('cumulative').print_stats(70); s.sort_stats('tottime').print_stats(40)" \
  "$RESEARCH_RUN_DIR/out/compose.prof" > "$RESEARCH_RUN_DIR/out/compose_prof.txt"
