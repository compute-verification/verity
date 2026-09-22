set -e
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
P=/workspace/tc/venv/bin/python
R=$RESEARCH_RUN_DIR
export PYTHONPATH=backends/gkr
echo "triton cache entries before: $(ls ~/.triton/cache 2>/dev/null | wc -l)"
rm -rf ~/.triton/cache
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
$P -c "import torch,triton;print('torch',torch.__version__,'triton',triton.__version__,torch.cuda.get_device_name(0))"
# --- statements / witnesses (only if missing); the built operand arrays live in /workspace/bb/instances-v1 (from run r20260922-022913-d462), seeds from the tree
mkdir -p /workspace/bb/instances-v1 && cp -rn fixtures/bench-instances/v1/seeds /workspace/bb/instances-v1/ 2>/dev/null || true
if [ ! -f /workspace/bb/pos4096/manifest.json ]; then
  uv run --frozen --package verity-numerical python -m gpu.bb_export export --root /workspace/bb/instances-v1 --tier vu-k1536 --hi 4096 --out /workspace/bb/pos4096 --procs 24
fi
if [ ! -f /workspace/bb/neg/manifest.json ]; then
  uv run --frozen --package verity-numerical python -m gpu.bb_export export --root /workspace/bb/instances-v1 --tier vu-k1536-neg --out /workspace/bb/neg --procs 24
fi
if [ ! -f /workspace/bb/synth2/manifest.json ]; then
  uv run --frozen --package verity-numerical python -m gpu.bb_export synth --out /workspace/bb/synth2 --vus 2 --steps 3
fi
# --- Rust verifier
(cd backends/gkr/verifier && CARGO_TARGET_DIR=/workspace/cargo-target-verifier cargo build --release 2>&1 | tail -2)
V=/workspace/cargo-target-verifier/release/verity-gkr-verify
# --- kernel path, clean cache
for cfg in "synth2:" "pos4096:--vus 64"; do
  D=/workspace/bb/${cfg%%:*}; A=${cfg#*:}; tag=$(basename $D)${A// /}
  echo "=== kernel path $D $A"
  $P -m gpu.run prove $D $A --device cuda --reps 2 --json $R/run_${tag}_kernel.json --proof-out $R/proof_${tag}_kernel.bin || echo "PYTHON VERIFY FAILED (kernel, $tag)"
  echo "--- rust verify"
  $V verify --dir $D --proof $R/proof_${tag}_kernel.bin $A --threads 16 --json $R/rv_${tag}_kernel.json || echo "RUST REJECTED (kernel, $tag)"
  echo "=== torch path $D $A"
  VERITY_GPU_TORCH_ONLY=1 $P -m gpu.run prove $D $A --device cuda --reps 1 --json $R/run_${tag}_torch.json --proof-out $R/proof_${tag}_torch.bin || echo "PYTHON VERIFY FAILED (torch, $tag)"
  $V verify --dir $D --proof $R/proof_${tag}_torch.bin $A --threads 16 --json $R/rv_${tag}_torch.json || echo "RUST REJECTED (torch, $tag)"
done
echo "triton cache entries after: $(ls ~/.triton/cache | wc -l)"
