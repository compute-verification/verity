#!/usr/bin/env bash
# corun.py on one leased GPU of vy-dc-h200 (cwd = a source tree with benchmarks/torch_gpu/sha512_rows beside s8_simple):
# the commit engine's build (nvcc) before the lease, then one lease of at most LEASE_MIN (default 10) minutes.
#
#   research run --on vy-dc-h200 --project verity --campaign e2e-demo --source . --cwd source \
#     --env GPU_LEASE_WHO=kernels-ks8simple -- bash benchmarks/torch_gpu/s8_simple/corun.sh [corun.py's options]
#
# Exit: 0 every co-run's words are the alone runs' and every fused kernel passes the PTX audit, 1 not, 3 the venv or the
# build failed, else the lease's code.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run under research run}; OUT=$D/out; mkdir -p "$OUT"
HERE=benchmarks/torch_gpu/s8_simple ROWS=benchmarks/torch_gpu/sha512_rows
for v in /workspace/jobs/venv312 /workspace/cp/venv312; do [ -z "${PY:-}" ] && [ -x "$v/bin/python3" ] && PY=$v/bin/python3; done
PY=${PY:-python3}
if ! "$PY" -c 'import torch, triton, pynvml' 2>/dev/null; then
  # a node without the jobs' venv (vy-dc-h200-3): check.sh's torch and Triton (2.13.0+cu129, 3.7.1) in scratch
  TMP=/workspace/tmp; [ -w "$TMP" ] || TMP=${TMPDIR:-/tmp}
  S=$(mktemp -d "$TMP/tmp-s8corun.XXXXXX"); trap 'rm -rf "$S"' EXIT
  export UV_CACHE_DIR=${UV_CACHE_DIR:-/workspace/cache/uv}
  { { uv venv -q -p 3.12 "$S/venv" || uv venv -q "$S/venv"; } && uv pip install -q -p "$S/venv/bin/python3" --index-url https://download.pytorch.org/whl/cu129 \
      --extra-index-url https://pypi.org/simple torch==2.13.0 numpy nvidia-ml-py; } > "$OUT/venv.txt" 2>&1 \
    || { tail -20 "$OUT/venv.txt"; echo "venv 3" | tee "$OUT/status.txt"; exit 3; }
  PY=$S/venv/bin/python3
fi
TV=$($PY -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null)
for c in "/usr/local/cuda-$TV" /usr/local/cuda /usr/local/cuda-13.0; do [ -x "$c/bin/nvcc" ] && { CUDA=$c; break; }; done
export PATH=${CUDA:-/usr/local/cuda}/bin:$PATH PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1
export SHA512_ROWS_BUILD=${SHA512_ROWS_BUILD:-$HOME/.cache/verity/sha512_rows}
git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}" > "$OUT/commit.txt"
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
case $ARCH in 90|100|120) ARCH=${ARCH}a ;; esac
PYTHONPATH=$PWD:$PWD/$ROWS "$PY" -c "import engine as E; print(E.device_lib('$ARCH').path)" > "$OUT/build.txt" 2>&1 \
  || { cat "$OUT/build.txt"; echo "build 3" | tee "$OUT/status.txt"; exit 3; }
gpu-lease status > "$OUT/leases-before.txt" 2>&1
echo "corun.sh: asking for one GPU at $(date -u +%H:%M:%SZ)"
export OUT HERE PY
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${LEASE_MIN:-10}" -- bash -c '
  echo "corun.sh: leased $GPU_LEASE_UUID at $(date -u +%H:%M:%SZ)"
  q=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,power.draw,power.limit,temperature.gpu
  nvidia-smi --query-gpu=$q --format=csv -i "$GPU_LEASE_UUID" > "$OUT/gpu.txt" 2>&1
  "$PY" "$HERE/corun.py" --out "$OUT" "$@"; rc=$?
  nvidia-smi --query-gpu=$q --format=csv,noheader -i "$GPU_LEASE_UUID" >> "$OUT/gpu.txt" 2>&1
  exit $rc' _ "$@"
rc=$?
echo "corun.sh: lease ended ($rc) at $(date -u +%H:%M:%SZ)"
echo "corun $rc" | tee "$OUT/status.txt"
cat "$OUT/summary.txt" 2>/dev/null
exit $rc
