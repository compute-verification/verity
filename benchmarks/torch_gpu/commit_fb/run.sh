#!/usr/bin/env bash
# The fallback commit call on the H200: check_roots.py (device roots equal the CPU references) and bench.py (GB/s at shape H,
# share of the step), under one gpu-lease. The CUDA library is built before the lease (nvcc only, no GPU).
#
#   research run --on vy-dc-h200 --project verity --campaign e2e-demo --source . --cwd source --timeout 30m \
#     --env GPU_LEASE_WHO=kernels-kcommitfb -- bash benchmarks/torch_gpu/commit_fb/run.sh [all|check|bench] [BENCH ARGS...]
#
# Env: VERITY_BENCH_VENV (default /workspace/cp/venv312: torch with CUDA, numpy), ARCH (default the GPU's compute
# capability, e.g. 90), LEASE_MIN (gpu-lease --max-min, default 20), GPU_LEASE_WHO (default kernels-kcommitfb).
set -euo pipefail
MODE=${1:-all}
shift || true
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
mkdir -p "$OUT"
PY=${VERITY_BENCH_VENV:-/workspace/cp/venv312}/bin/python
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONPATH=$ROOT:$ROOT/integrations/vllm:$ROOT/tools/research/src${PYTHONPATH:+:$PYTHONPATH}
ARCH=${ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)}
export GPU_LEASE_WHO=${GPU_LEASE_WHO:-kernels-kcommitfb}

"$PY" - "$ARCH" <<'EOF' | tee "$OUT/build.txt"
import sys
from verity_vllm.commit import row_tables_device as D
print("device", D.device_lib(sys.argv[1])._name)
print("host", D.host_lib()._name)
EOF

case "$MODE" in
  check) STEPS=("$PY $HERE/check_roots.py --out $OUT") ;;
  bench) STEPS=("$PY $HERE/bench.py --out $OUT $*") ;;
  all) STEPS=("$PY $HERE/check_roots.py --out $OUT" "$PY $HERE/bench.py --out $OUT $*") ;;
  *) echo "usage: run.sh [all|check|bench] [BENCH ARGS...]" >&2; exit 2 ;;
esac
CMD=$(printf '%s && ' "${STEPS[@]}")
gpu-lease 1 --wait --max-min "${LEASE_MIN:-20}" -- bash -c "nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv && ${CMD% && }"
