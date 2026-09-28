#!/usr/bin/env bash
# The POUS benchmark on a GPU pod, as a research Attempt:
#
#   git -C <tree> archive -o impl.tar <commit>        # the implementation's tree: #188's cursor/dense-decode-opt-6667
#   research run --on <pod> --project verity --source . --cwd source --send impl.tar --campaign pous \
#     -- bash benchmarks/pous/pod.sh IMPL=vllm-gpu IMPL_COMMIT=<commit> [WS=qwen2.5-0.5b/linears+head] [AUDITS=20]
#   research run --on <pod> --project verity --source . --cwd source --campaign pous \
#     -- bash benchmarks/pous/pod.sh IMPL=reference [WS=...] [MAX_SEGMENTS=N]
#
# vllm-gpu bootstraps the implementation tree's Python 3.12 environment (its ops/pod_bootstrap.sh --cpu: the venv, torch cu129,
# vLLM, the QWEN05 checkpoint; no native taps, which this benchmark does not use), then runs bench.py, which drives that tree's
# frozen harness (pous-dense-bench) and gates and profiles its result with this tree's verity_pous. reference runs
# bench.py's CPU reference on the pod's cores in a plain Python 3.12 venv.
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
REPO=$(pwd); I=$RESEARCH_RUN_DIR/inputs; O=$RESEARCH_RUN_DIR; W=${WORK:-/workspace}/pous-bench
IMPL=${IMPL:-vllm-gpu}; WS=${WS:-qwen2.5-0.5b/linears+head}; HW=${HW:-l40s-48gb/bf16}
SCHEME=${SCHEME:-$([ "$IMPL" = p2-gpu ] && echo p2-16448/v0 || echo band-chain/d12/v1)}
AUDITS=${AUDITS:-20}
mkdir -p "$W" "$O/bench"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader || true
nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null || true
command -v uv >/dev/null || { python3 -m pip install -q uv || exit 3; }
export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$REPO/packages/verity/src:$REPO/protocols/pous:$REPO/tools/research/src
if [ "$IMPL" = vllm-gpu ]; then
  [ -f "$I/impl.tar" ] || { echo "vllm-gpu needs --send impl.tar"; exit 2; }
  SHA=$(sha256sum "$I/impl.tar" | cut -c1-16)
  if [ "$(cat "$W/impl.sha" 2>/dev/null)" != "$SHA" ]; then
    rm -rf "$W/impl" && mkdir -p "$W/impl" && tar -xf "$I/impl.tar" -C "$W/impl" && echo "$SHA" > "$W/impl.sha" || exit 3
  fi
  (cd "$W/impl/integrations/vllm" && bash verity_vllm/ops/pod_bootstrap.sh --cpu --cases QWEN05 --out "$O/bootstrap") \
    > "$O/bootstrap.log" 2>&1 || { tail -30 "$O/bootstrap.log"; exit 3; }
  PY=${VENV:-/workspace/venv312}/bin/python
  export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} PATH=/usr/local/cuda/bin:$PATH
  # torch cu129 needs a CUDA 12.9 driver; below R575 a datacenter GPU runs it through NVIDIA's forward-compatibility libraries
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  if [ "$DRV" -lt 575 ]; then
    if [ ! -e /usr/local/cuda-12.9/compat/libcuda.so.1 ]; then
      . /etc/os-release; CUDA_REPO=ubuntu${VERSION_ID/./}
      dpkg -s cuda-keyring >/dev/null 2>&1 || { wget -q https://developer.download.nvidia.com/compute/cuda/repos/$CUDA_REPO/x86_64/cuda-keyring_1.1-1_all.deb && dpkg -i cuda-keyring_1.1-1_all.deb; } || exit 3
      export DEBIAN_FRONTEND=noninteractive
      apt-get update -qq && apt-get install -y -qq --no-install-recommends cuda-compat-12-9 >/dev/null || exit 3
    fi
    export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:${LD_LIBRARY_PATH:-}
  fi
  "$PY" -c 'import torch; assert torch.cuda.is_available(), "torch sees no CUDA device"; print(torch.__version__, torch.cuda.get_device_name(0))' || exit 3
  "$PY" -c 'import ninja' 2>/dev/null || uv pip install -q --python "$PY" ninja || exit 3   # torch's JIT build of pous_native
  export PATH=$(dirname "$PY"):$PATH
  "$PY" benchmarks/pous/bench.py --scheme "$SCHEME" --impl vllm-gpu --impl-tree "$W/impl" --impl-commit "${IMPL_COMMIT:-unknown}" \
    --build-dir "$W/pous_build" --python "$PY" --weight-set "$WS" --hardware "$HW" --out "$O/bench" --audits "$AUDITS"
elif [ "$IMPL" = p2-gpu ]; then
  # P2's kernels (benchmarks/pous/p2_gpu): CGBN at its pinned commit, libgmp for CGBN's host pass, nvcc for sm_89
  export PATH=/usr/local/cuda/bin:$PATH
  [ -f /usr/include/gmp.h ] || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libgmp-dev >/dev/null; } || exit 3
  CG=$W/CGBN
  [ -d "$CG/.git" ] || git clone -q https://github.com/NVlabs/CGBN "$CG" || exit 3
  git -C "$CG" checkout -q "$(cat benchmarks/pous/p2_gpu/CGBN.commit)" || exit 3
  make -C benchmarks/pous/p2_gpu CGBN="$CG" BIN="$W/p2bin" -j2 > "$O/p2-make.log" 2>&1 || { tail -20 "$O/p2-make.log"; exit 3; }
  cp "$W"/p2bin/*.ptxas "$O/" 2>/dev/null
  PY=$W/ref312/bin/python
  [ -x "$PY" ] || { uv venv -q "$W/ref312" --python 3.12 && uv pip install -q --python "$PY" numpy; } || exit 3
  "$PY" benchmarks/pous/bench.py --scheme "$SCHEME" --impl p2-gpu --p2-bin "$W/p2bin" --scratch "$W/scratch" \
    --impl-commit "${RESEARCH_SOURCE_COMMIT:-}" --weight-set "$WS" --hardware "$HW" --out "$O/bench" --audits "$AUDITS" \
    ${MAX_SEGMENTS:+--max-segments "$MAX_SEGMENTS"}
else
  PY=$W/ref312/bin/python
  [ -x "$PY" ] || { uv venv -q "$W/ref312" --python 3.12 && uv pip install -q --python "$PY" numpy; } || exit 3
  "$PY" benchmarks/pous/bench.py --scheme "$SCHEME" --impl reference --impl-commit "${RESEARCH_SOURCE_COMMIT:-}" \
    --weight-set "$WS" --hardware "$HW" --out "$O/bench" --audits "$AUDITS" ${MAX_SEGMENTS:+--max-segments "$MAX_SEGMENTS"}
fi
