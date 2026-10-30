#!/usr/bin/env bash
# kv_reads.py on the H200: m3draw's run again under its salt secret (~/.local/state/m3draw-1de3, 0600, never
# under $OUT), its roots held to inputs/roots.json (m3draw commit's), and drawn attention blocks of step 31 opened with
# the K and V rows they read at steps 0 .. 31, each against its own step's root. The card is leased for the replay only:
# the open (CPU, ~40 min) runs after the lease is released, its state beside the secret and removed at exit.
#   research run --on H200_NODE --project verity --source . --cwd source --timeout 7200 --send roots.json \
#     -- bash examples/h200/kv_reads_job.sh
set -euo pipefail
SECRET_DIR="$HOME/.local/state/m3draw-1de3"
SECRET="$SECRET_DIR/salt.secret"
PRIVATE="$SECRET_DIR/kv-$RESEARCH_RUN_ID"
trap 'rm -rf "$PRIVATE"' EXIT
uv sync -q --all-packages --extra torch
echo "nproc $(nproc)"; free -g | head -2
for c in /usr/local/cuda /usr/local/cuda-13.0 /usr/local/cuda-12.8 /usr/local/cuda-12.9; do [ -x "$c/bin/nvcc" ] && export CUDA=$c && break; done
echo "CUDA=${CUDA:-none}"; export PATH="${CUDA:-/usr/local/cuda}/bin:$PATH"
CUBIN_ONLY=1 bash benchmarks/torch_gpu/attn_ref/build.sh "$OUT/attn_ship" 2>&1 | tail -3
export ATTN_REF_CUBIN="$OUT/attn_ship/attn_ref.sm_90a.cubin"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
git rev-parse HEAD
KV=(uv run --no-sync python examples/h200/kv_reads.py --out "$OUT/kv" --secret "$SECRET" --private "$PRIVATE"
    --registered "$RESEARCH_RUN_DIR/inputs/roots.json" --slots "${SLOTS:-2}" --blocks "${BLOCKS:-8}"
    --shape h200 --seed demo --device cuda --batch 512 --steps 32 --prompt-bytes 8 --micro-batch 512
    --run-id e2e-h200-m3-b512-record)
gpu-lease 1 --wait --mem-gb 0 -- bash -c 'nvidia-smi --query-gpu=name,uuid,memory.used --format=csv && exec "$@"' _ \
  "${KV[@]}" --phase replay --workers "$(nproc)"
"${KV[@]}" --phase open --workers "$(( $(nproc) - 1 ))"
ls -la "$OUT/kv"
