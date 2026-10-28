#!/usr/bin/env bash
# Build the FA3 M1 hidden-value tap `verity_fa3_matReq.so` (export-and-commit tap of the sm90 FlashAttention-3
# forward, VERITY_TAP=3 / VERITY_MAT_SRC=59, layout v9 = the FA2 v8 word layout over PackGQA-packed rows) for THIS pod's H100 with
# the bf16 paged PackGQA forward kernels of head dims 64 and 128 in one .so.
#
#   cd <tree>/integrations/vllm && bash verity_vllm/ops/pod_fa3_tap.sh          # ~15-30 min first time (nvcc 12.4, sm_90a)
#   -> prints  FA3-TAP-OK <sha256> hdims=<mask> <path>   (or FA3-TAP-FAIL <reason>, exit 1)
#
# (1) pinned vllm-project/flash-attention @ 506341a1 (+ cutlass) at /workspace/cp/vllm-flash-attn (cloned if absent); (2) copied to
# $FA2/trees/fa3tap and patched by fa3_tap_src/apply_fa3_tap.py; (3) built by fa3_tap_src/build_fa3_ext.py into $FA2/build/fa3_matReq;
# (4) verified: loads, exposes fwd + verity_set_tap_v9 / verity_tap_hdims, mode 3 / src 59 / layout 9, hdims mask covers the request.
# Exactness is a separate step: tests/properties/fa_tap_exactness_gpu.py (needs the GPU).  Env: VENV (/workspace/venv-cu129), CUDA_HOME
# (first of /usr/local/cuda-12.9, /usr/local/cuda), FA2 (/workspace/cp/fa2), FA3_TAP_HDIMS (64,128), MAX_JOBS (32), TREE_ONLY=1,
# FA3_TAP_ROW_GUARD=1 (the guarded-max build `verity_fa3_matReqG.so` in $FA2/build/fa3_matReqG: the same tree with VERITY_ROW_GUARD=1,
# ROW word 3 = GuardNegInfZero(row_max) after a row's first key block, and VERITY_MAT_SRC=123, the MS class = each block's max * scale; neither in the default build).
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}; cd "$REPO"
VENV=${VENV:-/workspace/venv312}; PY=$VENV/bin/python   # the one Python 3.12 environment of verity_vllm/ops/pod_bootstrap.sh
if [ -z "${CUDA_HOME:-}" ]; then for c in /usr/local/cuda-12.9 /usr/local/cuda; do [ -x "$c/bin/nvcc" ] && { CUDA_HOME=$c; break; }; done; fi
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}; export PATH=$VENV/bin:$CUDA_HOME/bin:$PATH
export MAX_JOBS=${MAX_JOBS:-32}
FA2=${FA2:-/workspace/cp/fa2}; HDIMS=${FA3_TAP_HDIMS:-64,128}
PIN=506341a143fcabd4bb79052a7605ada727d6b3f5
SRC=$REPO/verity_vllm/acquire/fa3_tap_src
ROW_GUARD=${FA3_TAP_ROW_GUARD:-0}; NAME=verity_fa3_matReq; BUILD=fa3_matReq; DEFS="VERITY_TAP=3,VERITY_MAT_SRC=59"
[ "$ROW_GUARD" = "1" ] && { NAME=verity_fa3_matReqG; BUILD=fa3_matReqG; DEFS="VERITY_TAP=3,VERITY_MAT_SRC=123,VERITY_ROW_GUARD=1"; }
fail() { echo "FA3-TAP-FAIL $*"; exit 1; }
log() { echo "[fa3_tap] $(date -u +%FT%TZ) $*"; }
[ -x "$PY" ] || fail "no venv python at $PY"
command -v nvcc >/dev/null || fail "nvcc not found (CUDA_HOME=$CUDA_HOME)"
$PY -c "import ninja" 2>/dev/null || $PY -m pip install -q ninja || fail "ninja"
mkdir -p "$FA2/trees" "$FA2/build" "$FA2/logs"
if [ ! -d /workspace/cp/vllm-flash-attn/.git ]; then
  git clone -q https://github.com/vllm-project/flash-attention.git /workspace/cp/vllm-flash-attn || fail "clone vllm-flash-attn"
fi
( cd /workspace/cp/vllm-flash-attn && git checkout -q $PIN && git submodule update --init --depth 1 csrc/cutlass >/dev/null 2>&1 ) || fail "checkout $PIN"
[ "$(cd /workspace/cp/vllm-flash-attn && git rev-parse HEAD)" = "$PIN" ] || fail "tree is not at $PIN"
TREE=$FA2/trees/fa3tap
if [ ! -d "$TREE" ]; then cp -r /workspace/cp/vllm-flash-attn "$TREE" || fail "copy tree"; fi
python3 "$SRC/apply_fa3_tap.py" "$TREE/hopper" "$SRC" | tail -4 || fail "apply_fa3_tap"
grep -q "Verity_mat_ctx_sm90<kBlockM, kBlockN, kHeadDim>" "$TREE/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp" || fail "tree not patched"
[ "${TREE_ONLY:-0}" = "1" ] && { echo "FA3-TAP-TREE-OK $TREE"; exit 0; }
OUT=$FA2/build/$BUILD
NEED=$($PY -c "import sys; print(sum(1 << (int(h) // 32) for h in '$HDIMS'.split(',') if h))")
HAVE=0
if [ -f "$OUT/$NAME.so" ]; then
  HAVE=$($PY -c "import torch; torch.ops.load_library('$OUT/$NAME.so'); print(int(torch.ops.$NAME.verity_tap_hdims()))" 2>/dev/null || echo 0)
fi
# Source-set gate (same rule as pod_fa2_tap.sh): rebuild iff the fa3_tap_src files / pin / hdims / defines / nvcc changed, the .so is
# missing or lacks hdims, or FA3_TAP_FORCE=1; an existing hdims-complete build without a recorded hash is ADOPTED as the baseline (no rebuild).
NVCC_VER=$(nvcc --version 2>/dev/null | sed -n "s/.*release \([0-9.]*\), \(V[0-9.]*\).*/\1-\2/p" | head -1)   # e.g. 12.4-V12.4.131
SRC_SHA=$( (cd "$SRC" && find . -type f ! -name '*.pyc' ! -path '*/__pycache__/*' | LC_ALL=C sort | while read -r f; do printf '%s ' "$f"; sha256sum "$f" | cut -c1-64; done; echo "pin=$PIN hdims=$HDIMS defines=$DEFS nvcc=$NVCC_VER") | sha256sum | cut -c1-64)
OLD_SHA=$(cat "$OUT/source_set.sha256" 2>/dev/null || true)
WHY=""
if [ "${FA3_TAP_FORCE:-0}" = "1" ]; then WHY="FA3_TAP_FORCE=1"
elif [ ! -f "$OUT/$NAME.so" ]; then WHY="no .so at $OUT"
elif [ $(( HAVE & NEED )) -ne "$NEED" ]; then WHY="hdims mask $HAVE lacks $NEED ($HDIMS)"
elif [ -z "$OLD_SHA" ]; then log "ADOPT existing $OUT (hdims mask $HAVE covers $HDIMS) as source-set ${SRC_SHA:0:16} -- no recorded hash yet, no rebuild"; echo "$SRC_SHA" > "$OUT/source_set.sha256"
elif [ "$OLD_SHA" != "$SRC_SHA" ]; then WHY="source set / build config changed (${OLD_SHA:0:16} -> ${SRC_SHA:0:16})"
fi
if [ -n "$WHY" ]; then
  log "REBUILD $BUILD hdims=$HDIMS (nvcc $NVCC_VER) MAX_JOBS=$MAX_JOBS -- $WHY; source-set ${SRC_SHA:0:16}"
  NEW="$OUT.build-$$"; rm -rf "$NEW"; mkdir -p "$NEW"          # build beside the live .so, swap when done (see pod_fa2_tap.sh)
  BLOG=$FA2/logs/build_$BUILD.log
  PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" verity-vllm fa3-tap-build --tree "$TREE" --name "$NAME" --build-dir "$NEW" --hdims "$HDIMS" \
      --tree-commit "$PIN+fa3tap" $(for d in ${DEFS//,/ }; do printf -- '--define %s ' "$d"; done) > "$BLOG" 2>&1 \
      || { rm -rf "$NEW"; fail "build (see $BLOG: $(grep -a -m1 -i 'error' "$BLOG" | cut -c1-300))"; }
  echo "$SRC_SHA" > "$NEW/source_set.sha256"
  rm -rf "$OUT.prev"; [ -d "$OUT" ] && mv "$OUT" "$OUT.prev"; mv "$NEW" "$OUT"; rm -rf "$OUT.prev"
  log "$(tail -1 "$BLOG")"
elif [ -n "$OLD_SHA" ]; then
  log "REBUILD-SKIP $OUT: source-set ${SRC_SHA:0:16} == built ${OLD_SHA:0:16} (hdims mask $HAVE covers $HDIMS, nvcc $NVCC_VER)"
fi
$PY - "$OUT/$NAME.so" "$NEED" "$HDIMS" "$NAME" "$ROW_GUARD" <<'PYEOF' || fail "verify"
import sys, torch, hashlib, json, os
so, need, hdims, name, row_guard = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5])
torch.ops.load_library(so)
ns = getattr(torch.ops, name)
for op in ("fwd", "get_scheduler_metadata", "verity_set_tap_v9", "verity_set_tap_off", "verity_tap_skipped", "verity_tap_mode", "verity_tap_src", "verity_tap_hdims", "verity_tap_layout_version", "verity_tap_row3"):
    assert hasattr(ns, op), f"missing op {op}"
mode, src, have, ver = int(ns.verity_tap_mode()), int(ns.verity_tap_src()), int(ns.verity_tap_hdims()), int(ns.verity_tap_layout_version())
row3 = int(ns.verity_tap_row3())
assert mode == 3 and src == (123 if row_guard else 59), (mode, src, row_guard)
assert have & need == need, f"hdims mask {have} lacks {need}"
assert ver == 9, ver
assert row3 == row_guard, f"verity_tap_row3() = {row3}, the build asked for {row_guard}"
sha = hashlib.sha256(open(so, "rb").read()).hexdigest()
print(f"FA3-TAP-OK {sha} {so} hdims={hdims} mask={have} layout=v{ver} row3={row3}")
PYEOF
