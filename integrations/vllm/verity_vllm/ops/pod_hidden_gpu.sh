#!/usr/bin/env bash
# Build the GPU-resident SHA-256 chunk-tree extension `hidden_gpu_tree`
# that `commit_delta.py --gpu-tree` / `native_collect_v2b` import in every worker step (`native_host._gpu_ext_load` ->
# `import hidden_gpu` from `<repo>/out/gen/r9/cmt-hidden/src`).  The ship doesn't carry that directory, so without this step every
# Commit on a fresh pod fails with `ModuleNotFoundError: No module named 'hidden_gpu'`.
#
#   cd <tree>/integrations/vllm && bash verity_vllm/ops/pod_hidden_gpu.sh            # idempotent; ~1-2 min on first run, seconds after
#   -> prints  HIDDEN-GPU-OK <sha256 of hidden_gpu_tree.so>   (or HIDDEN-GPU-FAIL <reason>, exit 1)
#
# What it does: (1) materialises `out/gen/r9/cmt-hidden/src/{hidden_gpu.py,hidden_gpu_tree.cu}` from the shipped copies in
# `verity_vllm/commit/committer/hidden_gpu_src/` when that directory is absent;
# (2) JIT-builds the extension with torch.utils.cpp_extension.load for THIS pod's GPU (TORCH_CUDA_ARCH_LIST from the device: sm80/sm89/sm90)
# into the default torch-extensions cache -- the same directory the vLLM worker's `hidden_gpu.ext()` resolves when HIDDEN_GPU_BUILD is
# unset, so the worker only re-checks the ninja build (no compile) -- and also into $HIDDEN_GPU_BUILD if the caller sets it;
# (3) verifies the import + a GPU root check against the host hashlib reference (`hidden_gpu.py check`).
# Env: VENV (default /workspace/venv-cu129), CUDA_HOME (default /usr/local/cuda), REPO (default: this tree).
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}; cd "$REPO"
VENV=${VENV:-/workspace/venv312}; PY=$VENV/bin/python   # the one Python 3.12 environment of verity_vllm/ops/pod_bootstrap.sh
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}; export PATH=$VENV/bin:$CUDA_HOME/bin:$PATH
export VERITY_TREE=$REPO PYTHONPATH=$REPO:$REPO/../..:$REPO/../../tools/research/src:$REPO/out/gen/r9/cmt-hidden/src${PYTHONPATH:+:$PYTHONPATH}   # core + runner beside the integration
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
fail() { echo "HIDDEN-GPU-FAIL $*"; exit 1; }
[ -x "$PY" ] || fail "no venv python at $PY"
command -v nvcc >/dev/null || fail "nvcc not found (CUDA_HOME=$CUDA_HOME)"
$PY -c "import ninja" 2>/dev/null || $PY -m pip install -q ninja || fail "ninja"

# (1) the sources at the path native_host imports from
SRC=$REPO/out/gen/r9/cmt-hidden/src; SHIP=$REPO/verity_vllm/commit/committer/hidden_gpu_src
mkdir -p "$SRC"
# The shipped copy is the source of record: a stale copy at that path (an earlier ship's tar leaves it in place) is
# OVERWRITTEN when its bytes differ, never kept -- the fixed hidden_gpu.py (REPO from the importing tree) must be what native_host imports
for f in hidden_gpu.py hidden_gpu_tree.cu; do
  [ -s "$SHIP/$f" ] || fail "missing $SHIP/$f (re-ship the tree)"
  if [ ! -s "$SRC/$f" ]; then
    cp "$SHIP/$f" "$SRC/$f"; echo "[hidden_gpu] materialised $SRC/$f from $SHIP"
  elif ! cmp -s "$SHIP/$f" "$SRC/$f"; then
    cp "$SHIP/$f" "$SRC/$f"; echo "[hidden_gpu] REPLACED stale $SRC/$f with the shipped copy $SHIP/$f (bytes differed)"
  fi
done

# (2) build for this GPU: default torch-extensions cache (what the worker uses) + optional HIDDEN_GPU_BUILD
ARCH=$($PY -c "import torch; cc=torch.cuda.get_device_capability(0); print(f'{cc[0]}.{cc[1]}')") || fail "no CUDA device"
# Do NOT export TORCH_CUDA_ARCH_LIST here.  torch.utils.cpp_extension.load hashes the nvcc arch flags into the extension
# version: with the variable set to "8.9" it emits sm_89 only, unset (the vLLM worker / commit_delta / pod_ready import) it emits
# sm_89 + compute_89 PTX -> a different version -> every consumer would rebuild the cache this script just warmed (~70 s per workload
# start, and the readiness check's "loaded from cache" assertion fails).  Build exactly what the consumers build: same env.
# A caller that really wants a different arch list must set TORCH_CUDA_ARCH_LIST for the workers too (row_pod.sh / canary.sh env).
echo "[hidden_gpu] $(date -u +%FT%TZ) building hidden_gpu_tree for sm_${ARCH/./} (TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-<unset, as the worker>}) with $(nvcc --version | tail -1 | cut -c1-60) torch $($PY -c 'import torch;print(torch.__version__)')"
build_once() {   # $1 = HIDDEN_GPU_BUILD value ("" = default cache)
  HIDDEN_GPU_BUILD="$1" $PY - <<'EOF' || return 1
import os, sys, time
t0 = time.perf_counter()
import hidden_gpu as HG
e = HG.ext()
so = e.__file__ if hasattr(e, "__file__") else None
print(f"[hidden_gpu] built/loaded {so} in {time.perf_counter()-t0:.0f}s (HIDDEN_GPU_BUILD={os.environ.get('HIDDEN_GPU_BUILD') or '<default cache>'})", flush=True)
open("/tmp/hidden_gpu_so_path", "w").write(so or "")
EOF
}
build_once "" || fail "JIT build (default cache) failed"
SO=$(cat /tmp/hidden_gpu_so_path)
if [ -n "${HIDDEN_GPU_BUILD:-}" ]; then build_once "$HIDDEN_GPU_BUILD" || fail "JIT build into HIDDEN_GPU_BUILD=$HIDDEN_GPU_BUILD failed"; fi

# (3) verify: import from a fresh interpreter exactly as the worker does, and the GPU root == host hashlib root
$PY - <<'EOF' || fail "verification"
import json, os, sys, tempfile
import hidden_gpu as HG
e = HG.ext()
assert hasattr(e, "chunk_leaves") and hasattr(e, "tree_levels"), dir(e)
out = os.path.join(tempfile.gettempdir(), "hidden_gpu_check.json")
HG.main(HG.CheckOptions(out=out))      # the self-check: GPU chunk-tree root/levels/openings == host hashlib reference
r = json.load(open(out))
if not r.get("_all_ok"):
    raise SystemExit(f"GPU tree != host reference: {[c for c in r['cases'] if not (c['root_equal'] and c['levels_equal'] and c['openings_ok'])]}")
EOF
[ -s "$SO" ] || SO=$(ls -t "${HOME}/.cache/torch_extensions"/*/hidden_gpu_tree/hidden_gpu_tree.so 2>/dev/null | head -1)
[ -s "$SO" ] || fail "built .so not found"
echo "HIDDEN-GPU-OK $(sha256sum "$SO" | cut -c1-64) $SO sm_${ARCH/./}"
