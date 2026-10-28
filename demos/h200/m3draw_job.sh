#!/usr/bin/env bash
# m3draw.py on the H200: PHASE=commit writes the salt secret (~/.local/state/m3draw-1de3, 0600, never under
# $OUT) and the roots; PHASE=open re-runs, checks the roots against inputs/roots.json, opens inputs/draw.json's units (their
# values and salts, drawn.bin, beside the secret, never under $OUT) and checks them on CPU (m3check.py against
# inputs/registration.json, the coin's session in inputs/coin.tgz, inputs/challenger-key.json and inputs/main.txt).
#   research run --on H200_NODE --project verity --source . --cwd source --timeout 5400 --env PHASE=commit \
#     -- gpu-lease 1 --wait --mem-gb 0 -- bash demos/h200/m3draw_job.sh
set -euo pipefail
PHASE=${PHASE:?}
SECRET_DIR="$HOME/.local/state/m3draw-1de3"
uv sync -q --all-packages --extra torch
nvidia-smi --query-gpu=name,uuid,memory.total,memory.used --format=csv
echo "nproc $(nproc)"; free -g | head -2
for c in /usr/local/cuda /usr/local/cuda-13.0 /usr/local/cuda-12.8 /usr/local/cuda-12.9; do [ -x "$c/bin/nvcc" ] && export CUDA=$c && break; done
echo "CUDA=${CUDA:-none}"; export PATH="${CUDA:-/usr/local/cuda}/bin:$PATH"
CUBIN_ONLY=1 bash benchmarks/torch_gpu/attn_ref/build.sh "$OUT/attn_ship" 2>&1 | tail -3
export ATTN_REF_CUBIN="$OUT/attn_ship/attn_ref.sm_90a.cubin"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
git rev-parse HEAD
FLAGS=(--shape h200 --seed demo --device cuda --batch 512 --steps 32 --prompt-bytes 8 --micro-batch 512 --workers "$(nproc)"
       --run-id e2e-h200-m3-b512-record)
if [ "$PHASE" = commit ]; then
  mkdir -p "$SECRET_DIR"; chmod 700 "$SECRET_DIR"
  uv run --no-sync python demos/h200/m3draw.py commit --out "$OUT/m3" --secret "$SECRET_DIR/salt.secret" --new-secret "${FLAGS[@]}"
else
  PRIVATE="$SECRET_DIR/open-$RESEARCH_RUN_ID"
  uv run --no-sync python demos/h200/m3draw.py open --out "$OUT/m3" --secret "$SECRET_DIR/salt.secret" \
    --registered "$RESEARCH_RUN_DIR/inputs/roots.json" --draw "$RESEARCH_RUN_DIR/inputs/draw.json" --private "$PRIVATE" "${FLAGS[@]}"
  COIN=$(mktemp -d); tar -xzf "$RESEARCH_RUN_DIR/inputs/coin.tgz" -C "$COIN"
  uv run --no-sync python demos/h200/m3check.py --drawn "$OUT/m3" --payload "$PRIVATE/drawn.bin" \
    --registration "$RESEARCH_RUN_DIR/inputs/registration.json" --coin "$COIN/coin/coin.json" \
    --key-record "$RESEARCH_RUN_DIR/inputs/challenger-key.json" --main "$RESEARCH_RUN_DIR/inputs/main.txt" --out "$OUT/m3/check.json"
  ls -la "$PRIVATE"
fi
ls -la "$OUT/m3"
