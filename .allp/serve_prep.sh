set -euo pipefail
export PYTHONPATH=integrations/torch_loop
uv sync -q --all-packages --extra torch
git rev-parse HEAD
echo "nproc $(nproc)"; free -g | sed -n 2p; df -h /workspace | tail -1
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
D=/workspace/research/allp-h200
mkdir -p "$D"
T="$D/w-test.bin"
EP=
cleanup() { [ -n "$EP" ] && kill -TERM "$EP" 2>/dev/null && wait "$EP" 2>/dev/null; rm -f "$T"; df -h /workspace | tail -1; }
trap cleanup EXIT
S=text:verity/cmp92/h200/weights/v1
M="$D/weights-v1-h200.manifest.json"

t1=$(date +%s)
[ -s "$M" ] || { uv run --no-sync python -m verity_torch_loop.weights manifest --seed "$S" --digests --workers "$(nproc)" --out "$M" 2>&1 | tail -1; }
sha512sum "$M"

uv run --no-sync python -m verity_torch_loop.weights write --seed "$S" --workers "$(nproc)" --out "$T" 2>&1 | tail -1
sync
t2=$(date +%s); echo "write_s $((t2 - t1))"; ls -la "$T"; df -h /workspace | tail -1

for c in /usr/local/cuda /usr/local/cuda-13.0 /usr/local/cuda-12.8 /usr/local/cuda-12.9; do [ -x "$c/bin/nvcc" ] && export CUDA=$c && break; done
export PATH="${CUDA:-/usr/local/cuda}/bin:$PATH"
CUBIN_ONLY=1 bash benchmarks/torch_gpu/attn_ref/build.sh "$OUT/attn_ship" 2>&1 | tail -1
export ATTN_REF_CUBIN="$OUT/attn_ship/attn_ref.sm_90a.cubin" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export E2E_H200_SEED=demo E2E_H200_WEIGHTS_FILE="$T" E2E_H200_WEIGHTS_MANIFEST="$M" PYTHONPATH=benchmarks/e2e_h200:integrations/torch_loop
t3=$(date +%s)
uv run --no-sync python benchmarks/network_traces/demo_endpoint.py --out "$OUT/endpoint" --engine e2e_h200.engine:frozen \
  --host 127.0.0.1 --port 18780 --eos 0 --max-num-seqs 4 > "$OUT/endpoint.log" 2>&1 &
EP=$!
for _ in $(seq 1800); do grep -q '"listening"' "$OUT/endpoint.log" && break; kill -0 "$EP" || { tail -20 "$OUT/endpoint.log"; exit 4; }; sleep 1; done
echo "listening_after_s $(( $(date +%s) - t3 ))"; grep '"listening"' "$OUT/endpoint.log" | cut -c1-300
nvidia-smi --query-gpu=memory.used --format=csv,noheader
MODEL=$(curl -sS http://127.0.0.1:18780/v1/models | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
echo "model $MODEL"
curl -sS -N http://127.0.0.1:18780/v1/completions -H 'content-type: application/json' \
  -d "{\"model\": \"$MODEL\", \"prompt\": [72, 101, 108, 108, 111, 32, 72, 50], \"max_tokens\": 6, \"temperature\": 0.0, \"top_p\": 1.0, \"seed\": 1, \"stop\": [], \"stream\": true, \"user\": \"allp-smoke.00000\"}" \
  | cut -c1-200
kill -TERM "$EP"; wait "$EP" || true; EP=
cat "$OUT/endpoint/summary.json" 2>/dev/null | head -c 1500; echo
