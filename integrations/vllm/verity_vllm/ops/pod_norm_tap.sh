#!/usr/bin/env bash
# Build the norm-scale tap op `verity_norm_tap.so` for THIS pod's GPU (verity_vllm/acquire/norm_tap_src: vLLM d9105ea80's generic
# fused_add_rms_norm kernel with one store; build_ext.py says how), then check that it loads with its ops and tap version.
#
#   cd <tree>/integrations/vllm && bash verity_vllm/ops/pod_norm_tap.sh
#   -> the last line is NORM-TAP-OK {...} (or NORM-TAP-FAIL <reason>, exit 1)
#
# (1) fetches the three pinned vLLM sources build_ext.py reads (csrc/libtorch_stable/{layernorm_kernels.cu,cub_helpers.h,type_convert.cuh}
# at the pinned commit; build_ext refuses a file whose sha256 differs); (2) builds with `verity-vllm norm-tap-build` into $NT/build, the
# path config.RunEnv.norm_tap_so names, rebuilding only when the inputs changed (their sha256 beside the .so, source_set.sha256);
# (3) `verity-vllm norm-tap-build --check`.  Exactness is a separate step: tests/properties/norm_tap_exactness_gpu.py (needs the GPU).
# Env: VENV (default /workspace/venv312), CUDA_HOME (default: first of /usr/local/cuda-12.9, /usr/local/cuda), NT (default
# /workspace/cp/norm_tap), REPO (default: this tree), NORM_TAP_FORCE=1 (rebuild regardless).
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}; cd "$REPO"
VENV=${VENV:-/workspace/venv312}
if [ -z "${CUDA_HOME:-}" ]; then for c in /usr/local/cuda-12.9 /usr/local/cuda; do [ -x "$c/bin/nvcc" ] && { CUDA_HOME=$c; break; }; done; fi
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}; export PATH=$VENV/bin:$CUDA_HOME/bin:$PATH
NT=${NT:-/workspace/cp/norm_tap}
PIN=d9105ea8001e0a6d77a96327d17515bb5791fb36
SRC=$REPO/verity_vllm/acquire/norm_tap_src
VSRC=$NT/vllm/csrc/libtorch_stable
fail() { echo "NORM-TAP-FAIL $*"; exit 1; }
log() { echo "[norm_tap] $(date -u +%FT%TZ) $*"; }
command -v nvcc >/dev/null || fail "nvcc not found (CUDA_HOME=$CUDA_HOME)"
command -v verity-vllm >/dev/null || fail "no verity-vllm on PATH ($VENV/bin)"
command -v ninja >/dev/null || "$VENV/bin/pip" install -q ninja || fail "ninja"
mkdir -p "$VSRC" "$NT/build" "$NT/logs"

# (1) the pinned sources
for f in layernorm_kernels.cu cub_helpers.h type_convert.cuh; do
  [ -s "$VSRC/$f" ] || curl -fsSL "https://raw.githubusercontent.com/vllm-project/vllm/$PIN/csrc/libtorch_stable/$f" -o "$VSRC/$f" || fail "fetch $f"
done

# (2) build when the inputs changed
OUT=$NT/build
SET_SHA=$( (cd "$SRC" && find . -type f ! -name '*.pyc' ! -path '*/__pycache__/*' | LC_ALL=C sort | while read -r f; do printf '%s ' "$f"; sha256sum "$f" | cut -c1-64; done
            cd "$VSRC" && sha256sum layernorm_kernels.cu cub_helpers.h type_convert.cuh; nvcc --version | tail -1) | sha256sum | cut -c1-64)
if [ "${NORM_TAP_FORCE:-0}" = "1" ] || [ ! -f "$OUT/verity_norm_tap.so" ] || [ "$(cat "$OUT/source_set.sha256" 2>/dev/null)" != "$SET_SHA" ]; then
  log "BUILD $OUT (source set ${SET_SHA:0:16})"
  PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" verity-vllm norm-tap-build --vllm-src "$NT/vllm" --build-dir "$OUT" > "$NT/logs/build.log" 2>&1 \
    || fail "build (see $NT/logs/build.log: $(grep -a -m1 -i 'error' "$NT/logs/build.log" | cut -c1-200))"
  echo "$SET_SHA" > "$OUT/source_set.sha256"
  log "$(tail -1 "$NT/logs/build.log")"
else
  log "BUILD-SKIP $OUT: source set ${SET_SHA:0:16} unchanged"
fi

# (3) check
PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" verity-vllm norm-tap-build --check "$OUT/verity_norm_tap.so" || fail "check $OUT/verity_norm_tap.so"
