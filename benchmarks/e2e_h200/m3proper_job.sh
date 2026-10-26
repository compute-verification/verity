#!/usr/bin/env bash
# M3 proper on the H200 (CMP-161, PRF-172). PHASE=commit writes a new salt secret (~/.local/state/m3proper-1de3, 0600,
# never under $OUT, kept by `research keep`) and m3draw's roots.json for f3ca's prompts and flags. PHASE=replay re-runs
# under it with kv_reads' KeepingDriver for inputs/blocks.json's slots, holds every root to inputs/roots.json and leaves
# the drawn slots' rows beside the secret (m3bundle.py's input). The card is leased for the run only.
#   research run --on vy-dc-h200-2 --project verity --source . --cwd source --timeout 5400 --env PHASE=commit \
#     -- bash benchmarks/e2e_h200/m3proper_job.sh
set -euo pipefail
PHASE=${PHASE:?}
SECRET_DIR="$HOME/.local/state/m3proper-1de3"
SECRET="$SECRET_DIR/salt.secret"
uv sync -q --all-packages --extra torch
echo "nproc $(nproc)"; free -g | head -2
for c in /usr/local/cuda /usr/local/cuda-13.0 /usr/local/cuda-12.8 /usr/local/cuda-12.9; do [ -x "$c/bin/nvcc" ] && export CUDA=$c && break; done
echo "CUDA=${CUDA:-none}"; export PATH="${CUDA:-/usr/local/cuda}/bin:$PATH"
CUBIN_ONLY=1 bash benchmarks/torch_gpu/attn_ref/build.sh "$OUT/attn_ship" 2>&1 | tail -3
export ATTN_REF_CUBIN="$OUT/attn_ship/attn_ref.sm_90a.cubin"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
git rev-parse HEAD
FLAGS=(--shape h200 --seed demo --device cuda --batch 512 --steps 32 --prompt-bytes 8 --micro-batch 512
       --run-id e2e-h200-m3-b512-record)
LEASE=(gpu-lease 1 --wait --mem-gb 0 -- bash -c 'nvidia-smi --query-gpu=name,uuid,memory.used --format=csv && exec "$@"' _)
if [ "$PHASE" = commit ]; then
  mkdir -p "$SECRET_DIR"; chmod 700 "$SECRET_DIR"
  NEW=(--new-secret); [ -e "$SECRET" ] && NEW=()  # a secret an earlier attempt wrote and no run committed under
  "${LEASE[@]}" uv run --no-sync python benchmarks/e2e_h200/m3draw.py commit --out "$OUT/m3" --secret "$SECRET" "${NEW[@]}" \
    --workers "$(nproc)" "${FLAGS[@]}"
  stat -c '%a %s %n' "$SECRET"
  research keep --owner @compute-accounting --expires 14d --priority keep --root "$HOME" \
    --why "M3 proper's prover-held salt secret (CMP-161, run $RESEARCH_RUN_ID's roots)" .local/state/m3proper-1de3
else
  PRIVATE="$SECRET_DIR/replay-$RESEARCH_RUN_ID"
  "${LEASE[@]}" uv run --no-sync python benchmarks/e2e_h200/kv_reads.py --phase replay --out "$OUT/kv" --secret "$SECRET" \
    --private "$PRIVATE" --registered "$RESEARCH_RUN_DIR/inputs/roots.json" --drawn "$RESEARCH_RUN_DIR/inputs/blocks.json" \
    --workers "$(nproc)" "${FLAGS[@]}"
  ls -la "$PRIVATE"
fi
ls -la "$OUT"/*
