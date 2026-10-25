#!/usr/bin/env bash
# attn_bf16's one-GPU screen (the kernel-engineering skill's tier 2a) on an RTX PRO 6000 node, as one `research run`: the
# build, the SASS gate and fixture.py's check outside the lease, then run.py (the check, the control, the timing) under
# `gpu-lease 1`.  Writes $OUT/run.jsonl (run.py's rows), run.err (its summary), sass.json, ptxas.txt, attn_bf16.sass,
# fixture.json, toolchain.txt, source.txt, MANIFEST.sha256 and python.txt (the timing's torch).
#
#   research run --on vy-nebius-1 --cwd source -- env CHECK_SLOTS=/workspace/research/locks/gpu/slots \
#     python3 tools/verity/check/slot.py --by LANE --wait-max 3600 --run-max 2700 -- \
#     bash benchmarks/torch_gpu/attn_bf16/screen.sh [CASE ...]
#
# GPU_LEASE_WHO names the lease's holder (default kernels); PY the Python with CUDA torch for the timing's baselines (default
# the node's /workspace/jobs/venv312, else uv's, which times these kernels alone).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
: "${OUT:?OUT names the output directory of the run}"
export PATH=/usr/local/cuda/bin:$PATH GPU_LEASE_WHO=${GPU_LEASE_WHO:-kernels}
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
cd "$ROOT"
uv sync -q --all-packages --extra torch-cpu
bash "$HERE/build.sh" "$T/ship" "$@"
cp "$T/ship/"{sass.json,ptxas.txt,attn_bf16.sass,toolchain.txt,source.txt,MANIFEST.sha256} "$T/ship/check/fixture.json" "$OUT/"
PY=${PY:-/workspace/jobs/venv312/bin/python}
"$PY" -c 'import numpy' 2>/dev/null || PY="$(uv run which python)"
"$PY" -c 'import sys, numpy; print(sys.executable, numpy.__version__); import torch; print("torch", torch.__version__, torch.version.cuda)' > "$OUT/python.txt" 2>&1 || true
nvidia-smi --query-gpu=index,name,uuid,memory.used,clocks.sm --format=csv >> "$OUT/python.txt" || true
set +e
gpu-lease 1 --wait --max-min 30 -- timeout 25m "$PY" "$T/ship/run.py" "$T/ship/check" > "$OUT/run.jsonl" 2> "$OUT/run.err"
rc=$?
set -e
cat "$OUT/run.err" >&2
exit $rc
