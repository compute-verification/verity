#!/usr/bin/env bash
# served_passes.py on a node, from the run tree's root, in the harness's environment (benchmarks/harness/env.sh):
#   research run --on vy-nebius-2 --gpus 1 --mem-gb 96 --project verity --campaign pouw-served --cwd source -- \
#     bash benchmarks/pouw/pearl_c_vllm/served_passes.sh serve [served_passes.py serve's options]
#   research run --on vy-nebius-2 --cpus 48 --mem-gb 64 --project verity --campaign pouw-served --cwd source -- \
#     env PASSES=<the serve run's passes directory> bash benchmarks/pouw/pearl_c_vllm/served_passes.sh verify [--fast]
# serve builds the cubin (benchmarks/harness/setup/pearl_c.sh), gives $PASSES_ROOT/<run id> a retention record (owner
# @compute-accounting, expiring after KEEP_DAYS, 14 by default) and retains every pass of one served run there, outside the
# run directory, so custody uploads only served.json.  verify replays each pass that run retained, on the CPU.
set -euo pipefail
. benchmarks/harness/env.sh
OUT=${RESEARCH_RUN_DIR:?run this under research run}/out
mkdir -p "$OUT"
MODE=${1:?serve or verify}; shift
case "$MODE" in
  serve)
    . benchmarks/harness/setup/pearl_c.sh
    PASSES=${PASSES_ROOT:-$WORK/pouw/served-passes}/${RESEARCH_RUN_ID:?}
    mkdir -p "$PASSES"
    "$VENV/bin/python" -m research keep "$PASSES" --owner @compute-accounting --priority normal \
      --expires "$(date -u -d "+${KEEP_DAYS:-14} days" +%F)" --why "every retained pass of served run $RESEARCH_RUN_ID (CMP-70)" \
      > "$OUT/keep.json"
    "$VENV/bin/python" benchmarks/pouw/pearl_c_vllm/served_passes.py serve --passes "$PASSES" --out "$OUT/served.json" "$@" ;;
  verify)
    "$VENV/bin/python" benchmarks/pouw/pearl_c_vllm/served_passes.py verify --passes "${PASSES:?the serve run's passes directory}" \
      --out "$OUT/verify.json" "$@" ;;
  *) echo "served_passes.sh: serve or verify, not $MODE" >&2; exit 2 ;;
esac
