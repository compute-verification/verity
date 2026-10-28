#!/usr/bin/env bash
# M3 proper on the H200. PHASE=commit writes a new salt secret (~/.local/state/m3proper-1de3, 0600,
# never under $OUT, kept by `research keep`) and m3draw's roots.json for f3ca's prompts and flags. PHASE=replay re-runs
# under it with kv_reads' KeepingDriver for inputs/blocks.json's slots, holds every root to inputs/roots.json and leaves
# the drawn slots' rows beside the secret (m3bundle.py's input). The card is leased for the run only.
# PHASE=replay (inputs roots.json, step-registration.json, draw.json) keeps the drawn slots beside the secret; PHASE=build
# (receipt.json, weights-open.json besides) writes the step tree to $OUT/tree on the CPU (m3bundle.py).
#   research run --on H200_NODE --project verity --source . --cwd source --timeout 5400 --env PHASE=commit \
#     -- bash demos/h200/m3proper_job.sh
set -euo pipefail
PHASE=${PHASE:?}
SECRET_DIR="$HOME/.local/state/m3proper-1de3"
SECRET="$SECRET_DIR/salt.secret"
uv sync -q --all-packages --extra torch
echo "nproc $(nproc)"; free -g | head -2
for c in /usr/local/cuda /usr/local/cuda-13.0 /usr/local/cuda-12.8 /usr/local/cuda-12.9; do [ -x "$c/bin/nvcc" ] && export CUDA=$c && break; done
echo "CUDA=${CUDA:-none}"; export PATH="${CUDA:-/usr/local/cuda}/bin:$PATH"
CUBIN_ONLY=1 bash benchmarks/torch_gpu/attn_bf16/build.sh "$OUT/attn_ship" 2>&1 | tail -3
export ATTN_BF16_CUBIN="$OUT/attn_ship/attn_bf16.sm_90a.cubin"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
git rev-parse HEAD
FLAGS=(--shape h200 --seed demo --device cuda --batch 512 --steps 32 --prompt-bytes 8 --micro-batch 512
       --run-id e2e-h200-m3-b512-record)
LEASE=(gpu-lease 1 --wait --mem-gb 0 -- bash -c 'nvidia-smi --query-gpu=name,uuid,memory.used --format=csv && exec "$@"' _)
if [ "$PHASE" = commit ]; then
  mkdir -p "$SECRET_DIR"; chmod 700 "$SECRET_DIR"
  NEW=(--new-secret); [ -e "$SECRET" ] && NEW=()  # a secret an earlier attempt wrote and no run committed under
  "${LEASE[@]}" uv run --no-sync python demos/h200/m3draw.py commit --out "$OUT/m3" --secret "$SECRET" "${NEW[@]}" \
    --workers "$(nproc)" "${FLAGS[@]}"
  stat -c '%a %s %n' "$SECRET"
  research keep --owner @compute-accounting --expires 14d --priority keep --root "$HOME" \
    --why "M3 proper's prover-held salt secret (run $RESEARCH_RUN_ID's roots)" "$SECRET_DIR"
elif [ "$PHASE" = replay ]; then
  research keep --owner @compute-accounting --expires 14d --priority keep --root "$HOME" \
    --why "M3 proper's prover-held salt secret (the commit phase's roots)" "$SECRET_DIR" || true
  I="$RESEARCH_RUN_DIR/inputs"
  "${LEASE[@]}" uv run --no-sync python demos/h200/m3bundle.py replay --private "$SECRET_DIR/m3bundle" --secret "$SECRET" \
    --registered "$I/roots.json" --registration "$I/step-registration.json" --draw "$I/draw.json" --workers "$(nproc)" "${FLAGS[@]}"
  ls -la "$SECRET_DIR/m3bundle"
else
  I="$RESEARCH_RUN_DIR/inputs"
  uv run --no-sync python demos/h200/m3bundle.py build --private "$SECRET_DIR/m3bundle" --registration "$I/step-registration.json" \
    --draw "$I/draw.json" --receipt "$I/receipt.json" --weights "$I/weights-open.json" --out "$OUT/tree" --workers "$(nproc)"
fi
ls -la "$OUT"/*
