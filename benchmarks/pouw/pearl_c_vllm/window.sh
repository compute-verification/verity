#!/usr/bin/env bash
# The e2e's timed window on node 2, from the run tree's root, with GPU 1's build.sh output tarred as
# pearl-c-sm120-ship.tar:
#   research run --on vy-nebius-2 --project verity --campaign pouw --source <run tree> --cwd source \
#     --env GPU_LEASE_WHO=<you> --send pearl-c-sm120-ship.tar -- \
#     gpu-lease 8 --wait --timed --max-min 20 -- timeout 1140 bash benchmarks/pouw/pearl_c_vllm/window.sh
# It times on the first leased GPU only.  In order, the first failure ends the window:
#  1. identity: the device name and cc 12.0, with the UUID and clocks recorded;
#  2. the ship: its MANIFEST.sha256 holds and its run.py is the tree's;
#  3. GPU 1's device check on this cubin, v1 (G = 4) only, no timing (run.py check.json, SHAPES empty);
#  4. e2e.py: every mode's own gates, then its timing, then pearlc's verify pass, retained under $WORK/passes/<run id> (about
#     45 GB on node 2's /workspace, outside the run directory, so custody does not upload it).  Before e2e.py writes a
#     retained pass, its directory gets a retention record (`research keep`, priority keep, never expiring, owner
#     @compute-accounting; KEEP_PRIORITY, KEEP_EXPIRES and KEEP_OWNER override them), and the window fails KEEP without one.
# The reference replay runs after the window, on the CPU (verify.sh <run id>); its ACCEPT is replay-verified (development
# shortcut; the protocol's check is a ZK proof, PROTOCOL.md).  Needs setup.sh's $WORK.
# The venv's tools (ninja, which FlashInfer's JIT calls) come first on PATH, and FLASHINFER_CUDA_ARCH_LIST is setup.sh's, so
# FlashInfer finds the sampling module setup.sh built, with the same flags, in the build's own cache (flashinfer_env.sh; a
# ready.json whose library is elsewhere fails SETUP).  jit-built.txt lists every FlashInfer library
# built during the run: empty unless a first-call JIT landed in it.
# MODE=smoke (untimed, `gpu-lease 1`): the same checks, then `e2e.py --smoke` (both engines resident and each one's first
# sampling calls) into smoke.json; with RETAIN=1 the arm then serves e2e.py's SERVED requests, each a retained pass under
# $WORK/passes/<run id>.  MODE=profile (untimed, `gpu-lease 1`): the same checks, then `profile_decode.py` under
# nsys (the capture is its steady-state pass) into profile.json, with nsys's per-kernel sums merged in.  SHIP_TAR names a
# ship already on the node instead of the sent one.  MODE=perplexity (untimed, as node-2 fill or `gpu-lease 1`): the same
# checks, GPU 1's for both variants (v2 runs the unpromoted kernels), then `e2e.py --perplexity` into quality.json, a scheme
# group a chunk with CHUNK_MIN=0, exiting 99 while groups remain.  OUT names the output directory where there is no research
# run (a fill job).  SCHEDULE (serial, the default, or deferred) and SCHEME (the Pearl-C scheme, and so its hashing format)
# pass to e2e.py and profile_decode.py; the quality run names its own schemes.  GRAPHS=1 passes --graphs to e2e.py, FP8_GRAPHS=1 --fp8-graphs.
# WHOLE_STEP=1 passes --whole-step: the arm's decode forwards as vLLM's FULL graphs with Pearl-C's calls inside; with it,
# WHOLE_DEFER=1 passes --whole-defer: each call's tile hashing captured beside the next call; CAPTURE (sizes, comma-separated)
# passes --capture: the arm's vLLM graph sizes, the decode batch's alone by default.  BATCH passes --batch.
# STATEMENT (with WHOLE_STEP=1) passes --statement: hm96_rows.cu's cubin (a path, or "build" to build it here with
# /usr/local/cuda-13.0's nvcc), adding e2e.py's pearlc-zk mode, each decode forward's rows committed in the ZK statement's format too.
set -euo pipefail
WORK=${WORK:-/workspace/pouw/mvp-e2e}; PY=$WORK/venv312/bin/python
MODEL=${MODEL:-$WORK/models/llama-3.1-8b-instruct}
OUT=${OUT:-${RESEARCH_RUN_DIR:?run this under research run, or give OUT}}
mkdir -p "$OUT"
MODE=${MODE:-window}
SWITCHES=(${SCHEDULE:+--schedule "$SCHEDULE"} ${SCHEME:+--scheme "$SCHEME"} ${GRAPHS:+--graphs} ${FP8_GRAPHS:+--fp8-graphs}
          ${WHOLE_STEP:+--whole-step} ${WHOLE_DEFER:+--whole-defer} ${CAPTURE:+--capture "$CAPTURE"} ${BATCH:+--batch "$BATCH"})
fail() { echo "WINDOW_FAIL_$1: ${*:2}" | tee "$OUT/window.fail" >&2; exit 3; }
[ -x "$PY" ] && [ -f "$WORK/ready.json" ] || fail SETUP "no $WORK/ready.json: run setup.sh first"
export PATH=$WORK/venv312/bin:$PATH FLASHINFER_CUDA_ARCH_LIST=12.0f
# the epoch's coins are the Lean verifier of record's (e2e.EpochCoins, pouw.verifier.coins), which Lake builds and runs
command -v lake >/dev/null || export PATH="$HOME/.elan/bin:$PATH"
. benchmarks/pouw/pearl_c_vllm/flashinfer_env.sh
"$PY" -c 'import json, os, sys
p = json.load(open(sys.argv[1]))["flashinfer"]["prebuilt"]["sampling"]
sys.exit(not (p.startswith(os.environ["FLASHINFER_WORKSPACE_BASE"] + "/") and os.path.exists(p)))' "$WORK/ready.json" \
  || fail SETUP "ready.json's FlashInfer sampling library is not in this build's cache $FLASHINFER_WORKSPACE_BASE: run setup.sh"
export HF_HOME=${HF_HOME:-/workspace/hf}
export PYTHONPATH=$PWD:$PWD/tools/config:$PWD/integrations/vllm:$PWD/benchmarks/pouw
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} CUDA_DEVICE_ORDER=PCI_BUS_ID
# NVML ignores CUDA_VISIBLE_DEVICES, so every nvidia-smi query names the leased GPU by its UUID (gpu-lease's first)
UUID=${GPU_LEASE_UUID:-}; UUID=${UUID%%,*}
[ -n "$UUID" ] || fail IDENTITY "no GPU_LEASE_UUID: run under gpu-lease"
export E2E_GPU_UUID=$UUID
if [ "${STATEMENT:-}" = build ]; then
  /usr/local/cuda-13.0/bin/nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -cubin -o "$OUT/hm96_rows.cubin" \
    benchmarks/pouw/served_zk/hm96_rows.cu || fail STATEMENT "hm96_rows.cu did not build"
  STATEMENT=$OUT/hm96_rows.cubin
fi
[ -z "${STATEMENT:-}" ] || SWITCHES+=(--statement "$STATEMENT")
touch "$OUT/.started"
trap 'find "$FLASHINFER_WORKSPACE_BASE/.cache/flashinfer" -name "*.so" -newer "$OUT/.started" > "$OUT/jit-built.txt" 2>/dev/null || true' EXIT

nvidia-smi --query-gpu=uuid,name,compute_cap,clocks.sm,clocks.mem,driver_version --format=csv,noheader -i "$UUID" > "$OUT/identity.csv"
grep -q "RTX PRO 6000 Blackwell" "$OUT/identity.csv" && grep -q ", 12.0," "$OUT/identity.csv" || fail IDENTITY "$(cat "$OUT/identity.csv")"

mkdir -p "$OUT/ship" && tar -xf "${SHIP_TAR:-$OUT/inputs/pearl-c-sm120-ship.tar}" -C "$OUT/ship"
SHIP=$(dirname "$(find "$OUT/ship" -name MANIFEST.sha256 | head -1)")
(cd "$SHIP" && sha256sum -c --quiet MANIFEST.sha256) || fail SHIP "MANIFEST.sha256 does not hold"
cmp -s "$SHIP/run.py" benchmarks/pouw/pearl_c_sm120/run.py || fail SHIP "the ship's run.py is not the run tree's"

VARIANTS="sm120"; [ "$MODE" = perplexity ] && VARIANTS="sm120 sm120-unpromoted"
PEARLC_EXPECT_UUID=$UUID SHAPES="" VARIANTS=$VARIANTS "$PY" "$SHIP/run.py" "$SHIP/check/check.json" > "$OUT/gpu1-check.jsonl" \
  || fail GPU1_CHECK "run.py's device check failed (exit $?); see gpu1-check.jsonl"

keep_passes() {
  local d=$WORK/passes/${RESEARCH_RUN_ID:?}
  mkdir -p "$d" && PYTHONPATH=$PWD/tools/research/src python3 -m research keep "$d" --owner "${KEEP_OWNER:-@compute-accounting}" \
    --expires "${KEEP_EXPIRES:-never}" --priority "${KEEP_PRIORITY:-keep}" --why "retained passes of run $RESEARCH_RUN_ID" \
    ${RESEARCH_AGENT_URL:+--by "$RESEARCH_AGENT_URL"} > "$OUT/keep.json" || fail KEEP "no retention record on $d"
}
if [ "$MODE" = smoke ]; then
  [ -z "${RETAIN:-}" ] || keep_passes
  "$PY" benchmarks/pouw/pearl_c_vllm/e2e.py --model "$MODEL" --kernel benchmarks/pouw/pearl_c_sm120 --cubin "$SHIP/pearl_c_sm120.cubin" \
    --out "$OUT/smoke.json" --smoke ${RETAIN:+--retain "$WORK/passes/${RESEARCH_RUN_ID:?}"} "${SWITCHES[@]}"
elif [ "$MODE" = perplexity ]; then
  "$PY" benchmarks/pouw/pearl_c_vllm/e2e.py --model "$MODEL" --kernel benchmarks/pouw/pearl_c_sm120 --cubin "$SHIP/pearl_c_sm120.cubin" \
    --out "$OUT/quality.json" --perplexity --chunk-min "${CHUNK_MIN:-7}"
elif [ "$MODE" = profile ]; then
  nsys profile -o "$OUT/decode" -f true -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
    "$PY" benchmarks/pouw/pearl_c_vllm/profile_decode.py --model "$MODEL" --kernel benchmarks/pouw/pearl_c_sm120 \
    --cubin "$SHIP/pearl_c_sm120.cubin" --out "$OUT/profile.json" "${SWITCHES[@]}"
  nsys stats -q -f csv -r cuda_gpu_kern_sum -o "$OUT/decode" "$OUT/decode.nsys-rep"
  "$PY" benchmarks/pouw/pearl_c_vllm/profile_decode.py --nsys-kernels "$(ls "$OUT"/decode*cuda_gpu_kern_sum*.csv | head -1)" \
    --out "$OUT/profile.json"
else
  keep_passes
  "$PY" benchmarks/pouw/pearl_c_vllm/e2e.py --model "$MODEL" --kernel benchmarks/pouw/pearl_c_sm120 --cubin "$SHIP/pearl_c_sm120.cubin" \
    --out "$OUT/e2e.json" --retain "$WORK/passes/${RESEARCH_RUN_ID:?}" "${SWITCHES[@]}"
fi
