#!/usr/bin/env bash
# Build the router-softmax tap op `verity_router_tap.so` for THIS pod's GPU (verity_vllm/acquire/router_tap_src: vLLM d9105ea80's topkGating
# kernel with a Tap template parameter; build_ext.py says how), then check that it loads with its ops and tap version.
#
#   cd <tree>/integrations/vllm && bash verity_vllm/ops/pod_router_tap.sh
#   -> the last line is ROUTER-TAP-OK {...} (or ROUTER-TAP-FAIL <reason>, exit 1)
#
# (1) fetches the three pinned vLLM sources build_ext.py reads (csrc/libtorch_stable/moe/topk_softmax_kernels.cu, csrc/cuda_compat.h,
# csrc/libtorch_stable/cub_helpers.h at the pinned commit; build_ext refuses a file whose sha256 differs); (2) builds with
# `verity-vllm router-tap-build` into $RT/build, the path config.RunEnv.router_tap_so names, rebuilding only when the inputs changed (their
# sha256 beside the .so, source_set.sha256); (3) `verity-vllm router-tap-build --check`.  Exactness is a separate step:
# tests/properties/router_tap_exactness_gpu.py (needs the GPU).
# Env: VENV (default /workspace/venv312), CUDA_HOME (default: first of /usr/local/cuda-12.9, /usr/local/cuda), RT (default
# /workspace/cp/router_tap), REPO (default: this tree), ROUTER_TAP_FORCE=1 (rebuild regardless).
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}; cd "$REPO"
VENV=${VENV:-/workspace/venv312}
if [ -z "${CUDA_HOME:-}" ]; then for c in /usr/local/cuda-12.9 /usr/local/cuda; do [ -x "$c/bin/nvcc" ] && { CUDA_HOME=$c; break; }; done; fi
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}; export PATH=$VENV/bin:$CUDA_HOME/bin:$PATH
RT=${RT:-/workspace/cp/router_tap}
PIN=d9105ea8001e0a6d77a96327d17515bb5791fb36
SRC=$REPO/verity_vllm/acquire/router_tap_src
FILES="csrc/libtorch_stable/moe/topk_softmax_kernels.cu csrc/cuda_compat.h csrc/libtorch_stable/cub_helpers.h"
fail() { echo "ROUTER-TAP-FAIL $*"; exit 1; }
log() { echo "[router_tap] $(date -u +%FT%TZ) $*"; }
command -v nvcc >/dev/null || fail "nvcc not found (CUDA_HOME=$CUDA_HOME)"
command -v verity-vllm >/dev/null || fail "no verity-vllm on PATH ($VENV/bin)"
command -v ninja >/dev/null || "$VENV/bin/pip" install -q ninja || fail "ninja"
mkdir -p "$RT/build" "$RT/logs"

# (1) the pinned sources
for f in $FILES; do
  mkdir -p "$RT/vllm/$(dirname "$f")"
  [ -s "$RT/vllm/$f" ] || curl -fsSL "https://raw.githubusercontent.com/vllm-project/vllm/$PIN/$f" -o "$RT/vllm/$f" || fail "fetch $f"
done

# (2) build when the inputs changed
OUT=$RT/build
SET_SHA=$( (cd "$SRC" && find . -type f ! -name '*.pyc' ! -path '*/__pycache__/*' | LC_ALL=C sort | while read -r f; do printf '%s ' "$f"; sha256sum "$f" | cut -c1-64; done
            cd "$RT/vllm" && sha256sum $FILES; nvcc --version | tail -1) | sha256sum | cut -c1-64)
if [ "${ROUTER_TAP_FORCE:-0}" = "1" ] || [ ! -f "$OUT/verity_router_tap.so" ] || [ "$(cat "$OUT/source_set.sha256" 2>/dev/null)" != "$SET_SHA" ]; then
  log "BUILD $OUT (source set ${SET_SHA:0:16})"
  PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" verity-vllm router-tap-build --vllm-src "$RT/vllm" --build-dir "$OUT" > "$RT/logs/build.log" 2>&1 \
    || fail "build (see $RT/logs/build.log: $(grep -a -m1 -i 'error' "$RT/logs/build.log" | cut -c1-200))"
  echo "$SET_SHA" > "$OUT/source_set.sha256"
  log "$(tail -1 "$RT/logs/build.log")"
else
  log "BUILD-SKIP $OUT: source set ${SET_SHA:0:16} unchanged"
fi

# (3) check
PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" verity-vllm router-tap-build --check "$OUT/verity_router_tap.so" || fail "check $OUT/verity_router_tap.so"
