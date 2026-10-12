#!/usr/bin/env bash
# An untimed nsys profile of the served e2e (served_profile.py over the tree's own e2e.py: both engines resident, the reps
# interleaved, no verify passes), from a run tree's root, with GPU 1's ship already on node 2 (SHIP_TAR):
#   research run --on vy-nebius-2 --project verity --campaign pouw --source <run tree> --cwd source --env GPU_LEASE_WHO=<id> \
#     --env SHIP_TAR=<ship> [--env SCHEDULE=deferred] --send served_profile.sh --send served_profile.py -- \
#     bash "$RESEARCH_RUN_DIR"/inputs/served_profile.sh
# The GPU part runs in a one-GPU preemptible lease (window.sh's identity, ship and device checks first); the SQLite export and
# the analysis run after it, off the GPU.  NSYS=0 runs the same without the profiler (the diagnostic timing alone); OUT names
# the output directory (default the run's); GRAPHS=1 passes --graphs, WHOLE_STEP=1 --whole-step, ROWS_FORM=s --rows-form s; NSYS_ARGS adds to nsys profile (--cuda-graph-trace=node).
# WHOLE_DEFER=1 passes --whole-defer; BATCH and PROMPT the decode batch and its prompts' length (--batch, --prompt), MEM_BASE
# and MEM_ARM the engines' memory fractions (--mem-base, --mem-arm).  The ship is unpacked outside OUT, and removed with it.
set -euo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:?run this under research run}}; mkdir -p "$OUT"; export OUT
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:-/workspace/pouw/mvp-e2e}; PY=$WORK/venv312/bin/python
if [ "${1:-}" != gpu ]; then
  gpu-lease 1 --wait --preemptible --max-min "${MAX_MIN:-10}" -- bash "$0" gpu
  if [ "${NSYS:-1}" = 1 ]; then
    nsys export --type sqlite --force-overwrite true -o "$OUT/served.sqlite" "$OUT/served.nsys-rep"
    "$PY" "$HERE/served_profile.py" --analyze "$OUT/served.sqlite" --out "$OUT/served-profile.json"
    rm -f "$OUT/served.sqlite"
  fi
  exit 0
fi
MODEL=${MODEL:-$WORK/models/llama-3.1-8b-instruct}
export PATH=$WORK/venv312/bin:$PATH FLASHINFER_CUDA_ARCH_LIST=12.0f HF_HOME=${HF_HOME:-/workspace/hf}
. benchmarks/pouw/pearl_c_vllm/flashinfer_env.sh
export PYTHONPATH=$PWD:$PWD/integrations/vllm:$PWD/benchmarks/pouw:$PWD/benchmarks/pouw/pearl_c_vllm
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} CUDA_DEVICE_ORDER=PCI_BUS_ID SERVED_TREE=$PWD
UUID=${GPU_LEASE_UUID:?run under gpu-lease}; UUID=${UUID%%,*}; export E2E_GPU_UUID=$UUID
nvidia-smi --query-gpu=uuid,name,compute_cap,clocks.sm,clocks.mem,driver_version --format=csv,noheader -i "$UUID" > "$OUT/identity.csv"
grep -q "RTX PRO 6000 Blackwell" "$OUT/identity.csv" && grep -q ", 12.0," "$OUT/identity.csv" || { echo "identity"; exit 3; }
UNPACK=$(mktemp -d); trap 'rm -rf "$UNPACK"' EXIT
tar -xf "${SHIP_TAR:?}" -C "$UNPACK"
SHIP=$(dirname "$(find "$UNPACK" -name MANIFEST.sha256 | head -1)")
(cd "$SHIP" && sha256sum -c --quiet MANIFEST.sha256) && cmp -s "$SHIP/run.py" benchmarks/pouw/pearl_c_sm120/run.py || { echo "ship"; exit 3; }
PEARLC_EXPECT_UUID=$UUID SHAPES="" VARIANTS=sm120 "$PY" "$SHIP/run.py" "$SHIP/check/check.json" > "$OUT/gpu1-check.jsonl"
ARGS=(--model "$MODEL" --kernel benchmarks/pouw/pearl_c_sm120 --cubin "$SHIP/pearl_c_sm120.cubin" --out "$OUT/e2e.json"
      --reps "${REPS:-2}" ${SCHEDULE:+--schedule "$SCHEDULE"} ${SCHEME:+--scheme "$SCHEME"} ${GRAPHS:+--graphs} ${FP8_GRAPHS:+--fp8-graphs} ${WHOLE_STEP:+--whole-step}
      ${WHOLE_DEFER:+--whole-defer} ${ROWS_FORM:+--rows-form "$ROWS_FORM"} ${BATCH:+--batch "$BATCH"} ${PROMPT:+--prompt "$PROMPT"}
      ${MEM_BASE:+--mem-base "$MEM_BASE"} ${MEM_ARM:+--mem-arm "$MEM_ARM"})
if [ "${NSYS:-1}" = 1 ]; then
  nsys profile -o "$OUT/served" -f true -t cuda,nvtx -s none --cpuctxsw=none ${NSYS_ARGS:-} \
    "$PY" "$HERE/served_profile.py" -- "${ARGS[@]}"
else
  "$PY" "$HERE/served_profile.py" -- "${ARGS[@]}"
fi
