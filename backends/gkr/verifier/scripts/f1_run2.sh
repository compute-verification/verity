#!/bin/bash
# F1 triage on vy-sp1 (RTX 4090, clean Triton cache): committed kernel path vs torch path, Rust verifier on every proof,
# then the kernel/torch diff harness.  Exports exist from run 1 (/workspace/bb/{pos4096,neg,synth2}).
set -e
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
P=/workspace/tc/venv/bin/python
R=$RESEARCH_RUN_DIR
export PYTHONPATH=backends/gkr:backends/numerical/python:packages/verity/src
SIZES=${SIZES:-"synth2: pos4096:--vus_64 pos4096:--vus_512"}
echo "triton cache entries before: $(ls ~/.triton/cache 2>/dev/null | wc -l)"
rm -rf ~/.triton/cache
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
$P -c "import torch,triton;print('torch',torch.__version__,'triton',triton.__version__,torch.cuda.get_device_name(0))"
(cd backends/gkr/verifier && CARGO_TARGET_DIR=/workspace/cargo-target-verifier cargo build --release 2>&1 | tail -1)
V=/workspace/cargo-target-verifier/release/verity-gkr-verify
for cfg in $SIZES; do
  D=/workspace/bb/${cfg%%:*}; A=${cfg#*:}; A=${A//_/ }; tag=$(basename $D)${A// /}
  echo "=== kernel path $D $A"
  $P -m gpu.run prove $D $A --device cuda --reps 2 --json $R/run_${tag}_kernel.json --proof-out $R/proof_${tag}_kernel.bin || echo "PYTHON VERIFY FAILED (kernel, $tag)"
  echo "--- rust verify (kernel)"
  $V verify --dir $D --proof $R/proof_${tag}_kernel.bin $A --threads 16 --json $R/rv_${tag}_kernel.json || echo "RUST REJECTED (kernel, $tag)"
  echo "=== torch path $D $A"
  VERITY_GPU_TORCH_ONLY=1 $P -m gpu.run prove $D $A --device cuda --reps 1 --json $R/run_${tag}_torch.json --proof-out $R/proof_${tag}_torch.bin || echo "PYTHON VERIFY FAILED (torch, $tag)"
  echo "--- rust verify (torch)"
  $V verify --dir $D --proof $R/proof_${tag}_torch.bin $A --threads 16 --json $R/rv_${tag}_torch.json || echo "RUST REJECTED (torch, $tag)"
done
echo "=== diff harness (kernel vs torch, per level), B=64"
$P -m gpu.f1_diff /workspace/bb/pos4096 --vus 64 --json $R/diff_64.json || echo "DIFF HARNESS: MISMATCH"
echo "=== diff harness synth B=2"
$P -m gpu.f1_diff /workspace/bb/synth2 --json $R/diff_synth2.json || echo "DIFF HARNESS: MISMATCH"
echo "triton cache entries after: $(ls ~/.triton/cache | wc -l)"
