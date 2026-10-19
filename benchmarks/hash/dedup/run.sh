#!/usr/bin/env bash
# The dedup step (step.py) on one RTX PRO 6000 of node 2, from the run tree's root:
#   research run --on vy-nebius-2 --project verity --campaign hash-loop --source . --cwd source \
#     --env GPU_LEASE_WHO=kernels-dedup [--env ARGS='--layers 2 --reps 1'] -- bash benchmarks/hash/dedup/run.sh
# Pearl-C's ship is built first, off the GPU, into scratch.  Then one preemptible one-GPU lease of at most LEASE_MIN minutes
# runs the identity check, Pearl-C's own kernel check (the ship's check.json, bit for bit), and step.py: its checks, then the
# interleaved timing.  ARGS adds to step.py's arguments.  Outputs go to OUT (default the run's out/).
set -uo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:?run this under research run}/out}; mkdir -p "$OUT"; export OUT
WORK=${WORK:-/workspace/pouw/mvp-e2e}; PY=$WORK/venv312/bin/python; export WORK PY
if [ "${1:-}" != gpu ]; then
  : "${GPU_LEASE_WHO:?pass --env GPU_LEASE_WHO=<id>}"
  git rev-parse HEAD > "$OUT/tree.txt"
  S=$(mktemp -d /workspace/pouw/tmp-dedup.XXXXXX); trap 'rm -rf "$S"' EXIT
  echo "dedup: ship build at $(date -u +%H:%M:%SZ)"
  CUDA=${CUDA:-/usr/local/cuda-13.0} bash benchmarks/pouw/pearl_c_sm120/build.sh "$S/ship" > "$OUT/build.log" 2>&1 \
    || { tail -20 "$OUT/build.log"; exit 11; }
  (cd "$S/ship" && sha256sum pearl_c_sm120.cubin) | tee "$OUT/cubin.sha256"
  LEASE_MIN=${LEASE_MIN:-25}
  echo "dedup: lease at $(date -u +%H:%M:%SZ)"
  SHIP=$S/ship gpu-lease 1 --wait --preemptible --max-min "$LEASE_MIN" --mem-gb "${MEM_GB:-128}" -- \
    timeout -k 30 $((LEASE_MIN * 60 - 90)) bash "$0" gpu 2>&1 | tee "$OUT/gpu.log"
  rc=${PIPESTATUS[0]}
  echo "dedup: gpu part exited $rc at $(date -u +%H:%M:%SZ)"
  exit "$rc"
fi
export PYTHONPATH=$PWD:$PWD/integrations/vllm:$PWD/benchmarks/pouw
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} CUDA_DEVICE_ORDER=PCI_BUS_ID
UUID=${GPU_LEASE_UUID:?run under gpu-lease}; UUID=${UUID%%,*}; export E2E_GPU_UUID=$UUID
nvidia-smi --query-gpu=uuid,name,compute_cap,clocks.sm,clocks.mem,driver_version --format=csv,noheader -i "$UUID" > "$OUT/identity.csv"
grep -q "RTX PRO 6000 Blackwell" "$OUT/identity.csv" && grep -q ", 12.0," "$OUT/identity.csv" || { echo identity; exit 3; }
(cd "$SHIP" && sha256sum -c --quiet MANIFEST.sha256) && cmp -s "$SHIP/run.py" benchmarks/pouw/pearl_c_sm120/run.py || { echo ship; exit 3; }
PEARLC_EXPECT_UUID=$UUID SHAPES="" VARIANTS=sm120 "$PY" "$SHIP/run.py" "$SHIP/check/check.json" > "$OUT/gpu1-check.jsonl" \
  || { echo "Pearl-C's kernel check failed"; tail -3 "$OUT/gpu1-check.jsonl"; exit 4; }
echo "dedup: Pearl-C's kernel check passed at $(date -u +%H:%M:%SZ)"
# shellcheck disable=SC2086
"$PY" benchmarks/hash/dedup/step.py --kernel benchmarks/pouw/pearl_c_sm120 --cubin "$SHIP/pearl_c_sm120.cubin" \
  --out "$OUT/dedup.json" ${ARGS:-}
