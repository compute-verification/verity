#!/usr/bin/env bash
# Build the head-dim-portable FA2 M1 hidden-value tap `verity_fa2_matReq.so` (the export-and-commit
# tap, VERITY_TAP=3 / VERITY_MAT_SRC=59) for THIS pod's GPU with the bf16 split-KV kernels of head dims 64, 96, 128 and 256 in ONE .so,
# so Phi-3 (hd 96), Qwen2.5 / Mistral-7B / OLMoE / Qwen3 (hd 128) and Gemma-2 (hd 256) can bind the FA2 hidden family next to SmolLM2/Llama
# (hd 64).  256 is in the DEFAULT: on a tap built for 64,96,128 a gemma2-2b B1 Commit fails closed ('fa2 launch 0: head_dim 256 not ...',
# the manifest's hidden rows missing).  It is a default, not an env var at ship time, so no later ship without the variable rebuilds the
# tap back to three head dims (a build-config difference triggers a REBUILD both ways).
#
#   cd <tree>/integrations/vllm && bash verity_vllm/ops/pod_fa2_tap.sh                # ~10-15 min first time (nvcc), seconds after
#   -> prints  FA2-TAP-OK <sha256 of verity_fa2_matReq.so> hdims=<mask> <path>          (or FA2-TAP-FAIL <reason>, exit 1)
#
# What it does: (1) clones vllm-project/flash-attention at the pinned 506341a1 (+ cutlass submodule) into /workspace/cp/vllm-flash-attn
# when absent; (2) copies it to $FA2/trees/tap8 and applies verity_tap_v7.patch + apply_v8.py (parametric (kBlockN, kHeadDim) M1
# layout; see fa2_tap_src/apply_v8.py); (3) builds with `verity-vllm fa2-tap-build` (fa2_tap_src/build_ext.py; same flags as the pinned CMake build, sm arch from the
# device) into $FA2/build/matReq (the path row_pod.sh / commit_delta --hidden-so use; an earlier hd-64-only .so is kept as
# build/matReq_hd64_r9/); (4) verifies: the .so loads, exposes verity_set_tap_v8 / verity_tap_hdims, tap mode 3 / src 59, and the
# hdims mask contains every requested head dim.  Exactness per head dim is a separate step: tests/properties/fa_tap_exactness_gpu.py (needs the GPU).
# Env: VENV (default /workspace/venv-cu129), CUDA_HOME (default: first of /usr/local/cuda-12.9, /usr/local/cuda), FA2 (default
# /workspace/cp/fa2), FA2_TAP_HDIMS (alias HDIMS; default 64,96,128,256), FA2_TAP_ARCH (default: the device's compute capability, e.g. 8.9
# on L40S -- the arch the exactness evidence was produced with; the pinned vLLM _vllm_fa2_C ships sm_80 SASS, pass FA2_TAP_ARCH=8.0
# to mirror that instead and record it), MAX_JOBS (default 32), REPO (default: this tree), TREE_ONLY=1 (prepare tree, no build),
# FA2_TAP_ROW_GUARD=1 (the guarded-max build `verity_fa2_matReqG.so` in $FA2/build/matReqG: the same tree with VERITY_ROW_GUARD=1, ROW
# word 3 = GuardNegInfZero(row_max) after a row's first key block, and VERITY_MAT_SRC=123, the MS class = each block's max * scale; neither in the default build).
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}; cd "$REPO"
VENV=${VENV:-/workspace/venv312}; PY=$VENV/bin/python   # the one Python 3.12 environment of verity_vllm/ops/pod_bootstrap.sh
if [ -z "${CUDA_HOME:-}" ]; then for c in /usr/local/cuda-12.9 /usr/local/cuda; do [ -x "$c/bin/nvcc" ] && { CUDA_HOME=$c; break; }; done; fi
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}; export PATH=$VENV/bin:$CUDA_HOME/bin:$PATH
export MAX_JOBS=${MAX_JOBS:-32}
FA2=${FA2:-/workspace/cp/fa2}; HDIMS=${HDIMS:-${FA2_TAP_HDIMS:-64,96,128,256}}
PIN=506341a143fcabd4bb79052a7605ada727d6b3f5
SRC=$REPO/verity_vllm/acquire/fa2_tap_src
ROW_GUARD=${FA2_TAP_ROW_GUARD:-0}; NAME=verity_fa2_matReq; BUILD=matReq; DEFS="VERITY_TAP=3,VERITY_MAT_SRC=59"
[ "$ROW_GUARD" = "1" ] && { NAME=verity_fa2_matReqG; BUILD=matReqG; DEFS="VERITY_TAP=3,VERITY_MAT_SRC=123,VERITY_ROW_GUARD=1"; }
fail() { echo "FA2-TAP-FAIL $*"; exit 1; }
log() { echo "[fa2_tap] $(date -u +%FT%TZ) $*"; }
[ -x "$PY" ] || fail "no venv python at $PY"
command -v nvcc >/dev/null || fail "nvcc not found (CUDA_HOME=$CUDA_HOME)"
$PY -c "import ninja" 2>/dev/null || $PY -m pip install -q ninja || fail "ninja"
mkdir -p "$FA2/trees" "$FA2/build" "$FA2/logs"

# (1) pinned tree
if [ ! -d /workspace/cp/vllm-flash-attn/.git ]; then
  git clone -q https://github.com/vllm-project/flash-attention.git /workspace/cp/vllm-flash-attn || fail "clone vllm-flash-attn"
fi
( cd /workspace/cp/vllm-flash-attn && git checkout -q $PIN && git submodule update --init --depth 1 csrc/cutlass >/dev/null 2>&1 ) || fail "checkout $PIN"
[ "$(cd /workspace/cp/vllm-flash-attn && git rev-parse HEAD)" = "$PIN" ] || fail "tree is not at $PIN"

# (2) tap8 tree = pin + v7 patch + v8 (idempotent)
if [ ! -d "$FA2/trees/tap8" ]; then cp -r /workspace/cp/vllm-flash-attn "$FA2/trees/tap8" || fail "copy tree"; fi
( cd "$FA2/trees/tap8" && python3 "$SRC/apply_v8.py" csrc/flash_attn/src "$SRC" ) | tail -3 || fail "apply_v8"
grep -q "Verity_mat_ctx<kBlockM, kBlockN, kHeadDim>" "$FA2/trees/tap8/csrc/flash_attn/src/flash_fwd_kernel.h" || fail "tap8 tree not v8"
[ "${TREE_ONLY:-0}" = "1" ] && { echo "FA2-TAP-TREE-OK $FA2/trees/tap8"; exit 0; }

# (3) build for this GPU
ARCH=${FA2_TAP_ARCH:-$($PY -c "import torch; cc=torch.cuda.get_device_capability(0); print(f'{cc[0]}.{cc[1]}')")} || fail "no CUDA device"
[ -n "$ARCH" ] || fail "no CUDA device (set FA2_TAP_ARCH)"
OUT=$FA2/build/$BUILD
if [ "$ROW_GUARD" != "1" ] && [ -f "$OUT/verity_fa2_matReq.so" ] && ! $PY -c "import torch,sys; torch.ops.load_library('$OUT/verity_fa2_matReq.so'); sys.exit(0 if hasattr(torch.ops.verity_fa2_matReq,'verity_tap_hdims') else 1)" 2>/dev/null; then
  log "existing $OUT is the hd-64-only build -> kept as $FA2/build/matReq_hd64_r9"; rm -rf "$FA2/build/matReq_hd64_r9"; mv "$OUT" "$FA2/build/matReq_hd64_r9"
fi
NEED=$($PY - "$HDIMS" <<'EOF'
import sys; print(sum(1 << (int(h) // 32) for h in sys.argv[1].split(',') if h))
EOF
)
HAVE=0
if [ -f "$OUT/$NAME.so" ]; then
  HAVE=$($PY -c "import torch; torch.ops.load_library('$OUT/$NAME.so'); print(int(torch.ops.$NAME.verity_tap_hdims()))" 2>/dev/null || echo 0)
fi
# Rebuild only when the SOURCE SET or the BUILD CONFIG changed: sha256 over every file of fa2_tap_src (patch, apply_v8,
# build_ext, stubs, headers) + the pinned FA tree commit + arch + hdims + defines + the nvcc version, recorded beside the .so
# ($OUT/source_set.sha256) after a successful build.  Decision recorded in the release log: REBUILD (why) | REBUILD-SKIP (hash equal) |
# ADOPT (an existing hdims-complete build from before this gate is taken as the baseline for the current source set WITHOUT rebuilding -- the
# same assumption the hdims-only rule made; from then on any source/config change rebuilds).  FA2_TAP_FORCE=1 rebuilds regardless.
NVCC_VER=$(nvcc --version 2>/dev/null | sed -n "s/.*release \([0-9.]*\), \(V[0-9.]*\).*/\1-\2/p" | head -1)   # e.g. 12.4-V12.4.131
SRC_SHA=$( (cd "$SRC" && find . -type f ! -name '*.pyc' ! -path '*/__pycache__/*' | LC_ALL=C sort | while read -r f; do printf '%s ' "$f"; sha256sum "$f" | cut -c1-64; done; echo "pin=$PIN arch=$ARCH hdims=$HDIMS defines=$DEFS nvcc=$NVCC_VER") | sha256sum | cut -c1-64)
OLD_SHA=$(cat "$OUT/source_set.sha256" 2>/dev/null || true)
WHY=""
if [ "${FA2_TAP_FORCE:-0}" = "1" ]; then WHY="FA2_TAP_FORCE=1"
elif [ ! -f "$OUT/$NAME.so" ]; then WHY="no .so at $OUT"
elif [ $(( HAVE & NEED )) -ne "$NEED" ]; then WHY="hdims mask $HAVE lacks $NEED ($HDIMS)"
elif [ -z "$OLD_SHA" ]; then log "ADOPT existing $OUT (hdims mask $HAVE covers $HDIMS) as source-set ${SRC_SHA:0:16} -- no recorded hash yet, no rebuild"; echo "$SRC_SHA" > "$OUT/source_set.sha256"
elif [ "$OLD_SHA" != "$SRC_SHA" ]; then WHY="source set / build config changed (${OLD_SHA:0:16} -> ${SRC_SHA:0:16})"
fi
if [ -n "$WHY" ]; then
  log "REBUILD $BUILD hdims=$HDIMS arch=$ARCH (nvcc $NVCC_VER) MAX_JOBS=$MAX_JOBS -- $WHY; source-set ${SRC_SHA:0:16}"
  # build BESIDE the live .so and swap when done: the live tap stays loadable for a workload / a second release running on this pod
  # meanwhile (with two pod_release.sh at once, a shared build dir would be emptied mid-nvcc by the second's `rm -rf`)
  NEW="$OUT.build-$$"; rm -rf "$NEW"; mkdir -p "$NEW"
  BLOG=$FA2/logs/build_${BUILD}_v8.log
  PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" verity-vllm fa2-tap-build --tree "$FA2/trees/tap8/csrc" --name "$NAME" --build-dir "$NEW" --arch "$ARCH" --hdims "$HDIMS" \
      --tree-commit "$PIN+tap8" $(for d in ${DEFS//,/ }; do printf -- '--define %s ' "$d"; done) > "$BLOG" 2>&1 \
      || { rm -rf "$NEW"; fail "build (see $BLOG: $(grep -a -m1 -i 'error' "$BLOG" | cut -c1-200))"; }
  echo "$SRC_SHA" > "$NEW/source_set.sha256"
  rm -rf "$OUT.prev"; [ -d "$OUT" ] && mv "$OUT" "$OUT.prev"; mv "$NEW" "$OUT"; rm -rf "$OUT.prev"
  log "$(tail -1 "$BLOG")"
elif [ -n "$OLD_SHA" ]; then
  log "REBUILD-SKIP $OUT: source-set ${SRC_SHA:0:16} == built ${OLD_SHA:0:16} (hdims mask $HAVE covers $HDIMS, arch $ARCH, nvcc $NVCC_VER)"
fi

# (4) verify
$PY - "$OUT/$NAME.so" "$NEED" "$ARCH" "$HDIMS" "$NAME" "$ROW_GUARD" <<'EOF' || fail "verify"
import sys, torch, hashlib, json, os
so, need, arch, hdims, name, row_guard = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6])
torch.ops.load_library(so)
ns = getattr(torch.ops, name)
for op in ("varlen_fwd", "verity_set_tap", "verity_set_tap_v8", "verity_set_tap_tag", "verity_tap_skipped", "verity_tap_mode", "verity_tap_src", "verity_tap_hdims", "verity_tap_layout_version", "verity_tap_row3"):
    assert hasattr(ns, op), f"missing op {op}"
mode, src, have, ver = int(ns.verity_tap_mode()), int(ns.verity_tap_src()), int(ns.verity_tap_hdims()), int(ns.verity_tap_layout_version())
row3 = int(ns.verity_tap_row3())
assert mode == 3 and src == (123 if row_guard else 59), (mode, src, row_guard)
assert have & need == need, f"hdims mask {have} lacks {need}"
assert ver == 8, ver
assert row3 == row_guard, f"verity_tap_row3() = {row3}, the build asked for {row_guard}"
sha = hashlib.sha256(open(so, "rb").read()).hexdigest()
bi = os.path.join(os.path.dirname(so), "build_info.json")
info = json.load(open(bi)) if os.path.exists(bi) else {}
print(f"FA2-TAP-OK {sha} {so} arch={info.get('arch', arch)} hdims={hdims} mask={have} layout=v{ver} row3={row3}")
EOF
