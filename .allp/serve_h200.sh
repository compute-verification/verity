set -euo pipefail
W=/workspace/research/m2-load/weights-v1-h200.bin
M=/workspace/research/allp-h200/weights-v1-h200.manifest.json
SIZE=139736443904
git rev-parse HEAD; sha512sum "$M"
uv sync -q --all-packages --extra torch
for c in /usr/local/cuda /usr/local/cuda-13.0 /usr/local/cuda-12.8 /usr/local/cuda-12.9; do [ -x "$c/bin/nvcc" ] && export CUDA=$c && break; done
export PATH="${CUDA:-/usr/local/cuda}/bin:$PATH"
CUBIN_ONLY=1 bash benchmarks/torch_gpu/attn_ref/build.sh "$OUT/attn_ship" 2>&1 | tail -1
export ATTN_REF_CUBIN="$OUT/attn_ship/attn_ref.sm_90a.cubin" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export E2E_H200_SEED=demo E2E_H200_WEIGHTS_FILE="$W" E2E_H200_WEIGHTS_MANIFEST="$M"
export PYTHONPATH=benchmarks/e2e_h200:integrations/torch_loop

echo "$(date -u +%H:%M:%SZ) waiting for the load's W: $SIZE bytes, unchanged for 120 s"
while :; do
  if [ "$(stat -c %s "$W" 2>/dev/null || echo 0)" = "$SIZE" ]; then
    m=$(stat -c %Y "$W"); sleep 120
    [ "$(stat -c %Y "$W")" = "$m" ] && break
  else
    sleep 60
  fi
done
echo "$(date -u +%H:%M:%SZ) W is written"

for attempt in 1 2 3; do
  rc=0
  gpu-lease 1 --wait --mem-gb 0 -- uv run --no-sync python benchmarks/network_traces/demo_endpoint.py \
    --out /workspace/research/m2-serve/endpoint --engine e2e_h200.engine:frozen \
    --host 127.0.0.1 --port 18780 --eos 0 --max-num-seqs 4 || rc=$?
  echo "$(date -u +%H:%M:%SZ) endpoint exited $rc (attempt $attempt)"
  [ "$rc" = 0 ] && exit 0
  sleep 300
done
exit 1
