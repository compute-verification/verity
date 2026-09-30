#!/usr/bin/env bash
# The e2e's timed window on node 2 (research store docs/pouw/mvp-e2e.md, "RTX PRO 6000"), from the run tree's root, with
# GPU 1's build.sh output tarred as pearl-c-sm120-ship.tar:
#   research run --on vy-nebius-2 --project verity --campaign pouw --source <run tree> --cwd source \
#     --env GPU_LEASE_WHO=bc-dd22acf8 --send pearl-c-sm120-ship.tar -- \
#     gpu-lease 8 --wait --timed --max-min 20 -- timeout 1140 bash benchmarks/pouw/pearl_c_vllm/window.sh
# It times on the first leased GPU only.  In order, the first failure ends the window:
#  1. identity: the device name and cc 12.0, with the UUID and clocks recorded;
#  2. the ship: its MANIFEST.sha256 holds and its run.py is the tree's;
#  3. GPU 1's device check on this cubin, v1 (G = 4) only, no timing (run.py check.json, SHAPES empty);
#  4. e2e.py: every mode's own gates, then its timing, then pearlc's verify pass, retained under $WORK/passes/<run id> (about
#     45 GB on node 2's /workspace, outside the run directory, so custody does not upload it).
# The reference verifier runs after the window, on the CPU (verify.sh <run id>).  Needs setup.sh's $WORK.
set -euo pipefail
WORK=${WORK:-/workspace/pouw/mvp-e2e}; PY=$WORK/venv312/bin/python
MODEL=${MODEL:-$WORK/models/llama-3.1-8b-instruct}
OUT=${RESEARCH_RUN_DIR:?run this under research run}
fail() { echo "WINDOW_FAIL_$1: ${*:2}" | tee "$OUT/window.fail" >&2; exit 3; }
[ -x "$PY" ] && [ -f "$WORK/ready.json" ] || fail SETUP "no $WORK/ready.json: run setup.sh first"
export HF_HOME=${HF_HOME:-/workspace/hf}
export PYTHONPATH=$PWD/packages/verity/src:$PWD/protocols/pouw:$PWD/integrations/vllm:$PWD/benchmarks/pouw
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} CUDA_DEVICE_ORDER=PCI_BUS_ID

nvidia-smi --query-gpu=uuid,name,compute_cap,clocks.sm,clocks.mem,driver_version --format=csv,noheader -i 0 > "$OUT/identity.csv"
UUID=$(cut -d, -f1 "$OUT/identity.csv")
grep -q "RTX PRO 6000 Blackwell" "$OUT/identity.csv" && grep -q ", 12.0," "$OUT/identity.csv" || fail IDENTITY "$(cat "$OUT/identity.csv")"

mkdir -p "$OUT/ship" && tar -xf "$OUT/inputs/pearl-c-sm120-ship.tar" -C "$OUT/ship"
SHIP=$(dirname "$(find "$OUT/ship" -name MANIFEST.sha256 | head -1)")
(cd "$SHIP" && sha256sum -c --quiet MANIFEST.sha256) || fail SHIP "MANIFEST.sha256 does not hold"
cmp -s "$SHIP/run.py" benchmarks/pouw/pearl_c_sm120/run.py || fail SHIP "the ship's run.py is not the run tree's"

PEARLC_EXPECT_UUID=$UUID SHAPES="" VARIANTS=sm120 "$PY" "$SHIP/run.py" "$SHIP/check/check.json" > "$OUT/gpu1-check.jsonl" \
  || fail GPU1_CHECK "run.py's device check failed (exit $?); see gpu1-check.jsonl"

"$PY" benchmarks/pouw/pearl_c_vllm/e2e.py --model "$MODEL" --kernel benchmarks/pouw/pearl_c_sm120 --cubin "$SHIP/pearl_c_sm120.cubin" \
  --out "$OUT/e2e.json" --retain "$WORK/passes/${RESEARCH_RUN_ID:?}"
