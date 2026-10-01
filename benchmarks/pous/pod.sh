#!/usr/bin/env bash
# The POUS benchmark on a GPU pod, as a research Attempt:
#
#   git -C <tree> archive -o impl.tar <commit>        # the implementation's tree: #188's cursor/dense-decode-opt-6667
#   research run --on <pod> --project verity --source . --cwd source --send impl.tar --campaign pous \
#     -- bash benchmarks/pous/pod.sh IMPL=vllm-gpu IMPL_COMMIT=<commit> [WS=qwen2.5-0.5b/linears+head] [AUDITS=20]
#   research run --on <pod> --project verity --source . --cwd source --campaign pous \
#     -- bash benchmarks/pous/pod.sh IMPL=reference [WS=...] [MAX_SEGMENTS=N]
#   ... -- bash benchmarks/pous/pod.sh IMPL=p2-gpu [WS=...]      # built for the device's arch (ARCH=sm_NN overrides)
#   ... -- bash benchmarks/pous/pod.sh IMPL=band-gpu [WS=...] [NVME_DIR=/workspace/nvme] [SCHEME=dense-chain/v1]
#   ... -- bash benchmarks/pous/pod.sh IMPL=p2-v1-gpu [WS=...] [NVME_DIR=...]       # P2 v1 (builds p2_v1/libp2v1.so with CGBN)
#   ... -- bash benchmarks/pous/pod.sh IMPL=side-by-side [WS=...] [NVME_DIR=...]    # band-gpu then p2-v1-gpu, and the table
#   ... -- bash benchmarks/pous/pod.sh IMPL=side-by-side GATES_ONLY=1 MAX_SEGMENTS=2  # build check: kernels built, gates run
#   ... -- bash benchmarks/pous/pod.sh IMPL=band-gpu BAND_D=6 [WS=...]   # the band at in-degree d (>= 5; side-by-side's band arm too)
#
# GATES_ONLY=1 (band-gpu, p2-v1-gpu, side-by-side) builds the kernels for the device, runs the kernel gates (the pinned vector
# on the device, the reference segments or blocks, the round trip) and stops: nothing is timed and nothing is published.
#
# HW, the census id recorded with the result, is the pod's GPU: the H100 SXM, the L40S, the RTX 4090 and the RTX PRO 6000
# Blackwell Server Edition are known, and any other GPU, or none, needs HW=<census/hardware.json id>.
#
# vllm-gpu bootstraps the implementation tree's Python 3.12 environment (its ops/pod_bootstrap.sh --cpu: the venv, torch cu129,
# vLLM, the QWEN05 checkpoint; no native taps, which this benchmark does not use), then runs bench.py, which drives that tree's
# frozen harness (pous-dense-bench) and gates and profiles its result with this tree's verity_pous. reference runs
# bench.py's CPU reference on the pod's cores in a plain Python 3.12 venv. band-gpu needs no implementation tree: a Python 3.12
# venv with torch built for the pod's CUDA toolkit (TORCH_CUDA=cuNNN overrides), numpy and ninja, and this tree's band_gpu/.
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
REPO=$(pwd); I=$RESEARCH_RUN_DIR/inputs; O=$RESEARCH_RUN_DIR; W=${WORK:-/workspace}/pous-bench
IMPL=${IMPL:-vllm-gpu}; WS=${WS:-qwen2.5-0.5b/linears+head}
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$GPU" in                                            # the census id is the device's, never a default another GPU inherits
  *"H100 80GB HBM3"*) HW=${HW:-h100-sxm5-80gb/bf16} ;;
  *L40S*) HW=${HW:-l40s-48gb/bf16} ;;
  *"RTX 4090"*) HW=${HW:-rtx-4090/e4m3} ;;                 # the census's RTX 4090 row (it has no bf16 row yet)
  *"RTX PRO 6000 Blackwell Server Edition"*) HW=${HW:-rtx-pro-6000-server/bf16} ;;  # sm_120; other RTX PRO 6000s need HW=
  *) [ -n "${HW:-}" ] || { echo "no census id for GPU '$GPU': pass HW=<census/hardware.json id>"; exit 2; } ;;
esac
SCHEME=${SCHEME:-$([ "$IMPL" = p2-gpu ] && echo p2-16448/v2 || echo band-chain/d12/v1)}
AUDITS=${AUDITS:-20}
GO=$([ "${GATES_ONLY:-0}" = 1 ] && echo --gates-only)
mkdir -p "$W" "$O/bench"
nvcc_release() { "$1/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1 \2/p'; }
# A CUDA 12 toolkit for torch's JIT build (torch's wheels are CUDA 12; a CUDA 13 nvcc does not match them), at least 12.8 on a
# Blackwell GPU (sm_100, sm_120): CUDA_HOME if it is one, else an installed /usr/local/cuda-12.x, else 12.8 from NVIDIA's apt
# repository (the Nebius image ships CUDA 13.0)
cuda12_home() {
  local need=0 cc home major minor
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
  [ "${cc%%.*}" -ge 10 ] 2>/dev/null && need=8
  for home in ${CUDA_HOME:-/usr/local/cuda} /usr/local/cuda-12.9 /usr/local/cuda-12.8 /usr/local/cuda; do
    read -r major minor <<< "$(nvcc_release "$home")"
    [ "${major:-0}" = 12 ] && [ "${minor:-0}" -ge "$need" ] && { echo "$home"; return 0; }
  done
  if [ ! -x /usr/local/cuda-12.8/bin/nvcc ]; then
    . /etc/os-release; local repo=ubuntu${VERSION_ID/./}
    dpkg -s cuda-keyring >/dev/null 2>&1 || { wget -q "https://developer.download.nvidia.com/compute/cuda/repos/$repo/x86_64/cuda-keyring_1.1-1_all.deb" \
      && dpkg -i cuda-keyring_1.1-1_all.deb >/dev/null; } || return 3
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends cuda-nvcc-12-8 \
      cuda-cudart-dev-12-8 cuda-cccl-12-8 >/dev/null || return 3
  fi
  echo /usr/local/cuda-12.8
}
# band-gpu's Python: torch pinned (TORCH_VERSION) from the PyTorch index for the CUDA 12 toolkit (cu126 below 12.8, cu128 from it)
gpu_env() {
  local home major minor
  home=$(cuda12_home) || { echo "no CUDA 12 toolkit for this GPU, and none could be installed"; return 3; }
  export CUDA_HOME=$home PATH=$home/bin:$PATH
  read -r major minor <<< "$(nvcc_release "$CUDA_HOME")"
  TORCH_CUDA=${TORCH_CUDA:-$([ "${minor:-8}" -ge 8 ] && echo cu128 || echo cu126)}
  TORCH_VERSION=${TORCH_VERSION:-2.7.1}
  local venv=$W/gpu312-torch$TORCH_VERSION-$TORCH_CUDA
  PY=$venv/bin/python
  [ -x "$PY" ] || { uv venv -q "$venv" --python 3.12 && uv pip install -q --python "$PY" numpy ninja \
    && uv pip install -q --python "$PY" "torch==$TORCH_VERSION" --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"; } || return 3
  "$PY" -c 'import torch; assert torch.cuda.is_available(), "torch sees no CUDA device"; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))' || return 3
  export PATH=$(dirname "$PY"):$PATH
}
# the host facts a result is re-derived with, and two distinct physical cores (not SMT siblings) for the verifier and responder
record_host() {
  nvidia-smi -q -d CLOCK,PERFORMANCE,POWER > "$O/clocks-before.txt" 2>&1 || true
  nvcc --version > "$O/nvcc.txt" 2>&1 || true
  lscpu -e > "$O/lscpu.txt" 2>&1 || true
  uv pip freeze --python "$PY" > "$O/pip-freeze.txt" 2>/dev/null || true
  if [ -n "${NVME_DIR:-}" ]; then
    mkdir -p "$NVME_DIR"; findmnt -T "$NVME_DIR" > "$O/nvme-mount.txt" 2>&1; lsblk -o NAME,MODEL,TRAN,ROTA,SIZE > "$O/lsblk.txt" 2>&1
  fi
  read -r VCPU RCPU <<< "$(lscpu -p=CPU,CORE 2>/dev/null | grep -v '^#' | awk -F, '!seen[$2]++ {print $1}' | sed -n '3p;4p' | tr '\n' ' ')"
  VCPU=${VCPU:-3}; RCPU=${RCPU:-2}
}
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
elif [ "$IMPL" = band-gpu ]; then
  gpu_env || exit 3
  record_host
  "$PY" benchmarks/pous/bench.py --scheme "$SCHEME" --impl band-gpu --build-dir "$W/pous_build" --impl-commit "${RESEARCH_SOURCE_COMMIT:-}" \
    --weight-set "$WS" --hardware "$HW" --out "$O/bench" --audits "$AUDITS" --workers "$(nproc)" --verifier-cpu "$VCPU" \
    --responder-cpu "$RCPU" ${NVME_DIR:+--nvme-dir "$NVME_DIR"} ${MAX_SEGMENTS:+--max-segments "$MAX_SEGMENTS"} $GO \
    ${BAND_D:+--band-d "$BAND_D"}
  RC=$?
  nvidia-smi -q -d CLOCK,PERFORMANCE,POWER > "$O/clocks-after.txt" 2>&1 || true
  exit $RC
elif [ "$IMPL" = p2-v1-gpu ] || [ "$IMPL" = side-by-side ]; then
  gpu_env || exit 3
  record_host
  # P2 v1's library (benchmarks/pous/p2_v1): CGBN at p2_gpu's pinned commit, libgmp for CGBN's host pass, the device's arch
  ARCH=${ARCH:-sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)}
  echo '#include <gmp.h>' | cpp >/dev/null 2>&1 || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libgmp-dev >/dev/null; } || exit 3
  CG=$W/CGBN
  [ -d "$CG/.git" ] || git clone -q https://github.com/NVlabs/CGBN "$CG" || exit 3
  git -C "$CG" checkout -q "$(cat benchmarks/pous/p2_gpu/CGBN.commit)" || exit 3
  P2LIB=$W/p2v1bin-$ARCH
  make -C benchmarks/pous/p2_v1 CGBN="$CG" BIN="$P2LIB" ARCH="$ARCH" > "$O/p2v1-make.log" 2>&1 || { tail -20 "$O/p2v1-make.log"; exit 3; }
  cp "$P2LIB"/*.ptxas "$O/" 2>/dev/null
  arm() {  # arm IMPL OUT [sub]: one bench.py run of that implementation into OUT; a sub-run writes no result.json of its own
    ${3:+env -u RESEARCH_RUN_DIR} "$PY" benchmarks/pous/bench.py --scheme "$SCHEME" --impl "$1" --build-dir "$W/pous_build" --p2-lib "$P2LIB" \
      --impl-commit "${RESEARCH_SOURCE_COMMIT:-}" --weight-set "$WS" --hardware "$HW" --out "$2" --audits "$AUDITS" \
      --workers "$(nproc)" --verifier-cpu "$VCPU" --responder-cpu "$RCPU" ${NVME_DIR:+--nvme-dir "$NVME_DIR"} \
      ${MAX_SEGMENTS:+--max-segments "$MAX_SEGMENTS"} $GO $([ "$1" = band-gpu ] && [ -n "${BAND_D:-}" ] && echo --band-d "$BAND_D")
  }
  if [ "$IMPL" = side-by-side ]; then
    mkdir -p "$O/bench/band" "$O/bench/p2"
    arm band-gpu "$O/bench/band" sub; RB=$?
    nvidia-smi -q -d CLOCK,PERFORMANCE,POWER > "$O/clocks-between.txt" 2>&1 || true
    arm p2-v1-gpu "$O/bench/p2" sub; RP=$?
    if [ -n "$GO" ]; then   # gates only: no measurements to table
      RC=0
    else
      "$PY" benchmarks/pous/side_by_side.py "$O/bench/band/bench-result.json" "$O/bench/p2/bench-result.json" \
        --out "$O/bench/side-by-side.json" --markdown "$O/bench/side-by-side.md"
      RC=$?
    fi
    [ $RB -ne 0 ] || [ $RP -ne 0 ] && RC=1
  else
    arm p2-v1-gpu "$O/bench"; RC=$?
  fi
  nvidia-smi -q -d CLOCK,PERFORMANCE,POWER > "$O/clocks-after.txt" 2>&1 || true
  exit $RC
elif [ "$IMPL" = p2-gpu ]; then
  # P2's kernels (benchmarks/pous/p2_gpu): CGBN at its pinned commit, libgmp for CGBN's host pass, nvcc for the device's arch
  export PATH=/usr/local/cuda/bin:$PATH
  ARCH=${ARCH:-sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)}
  echo '#include <gmp.h>' | cpp >/dev/null 2>&1 || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libgmp-dev >/dev/null; } || exit 3
  CG=$W/CGBN
  [ -d "$CG/.git" ] || git clone -q https://github.com/NVlabs/CGBN "$CG" || exit 3
  git -C "$CG" checkout -q "$(cat benchmarks/pous/p2_gpu/CGBN.commit)" || exit 3
  BIN=$W/p2bin-$ARCH
  make -C benchmarks/pous/p2_gpu CGBN="$CG" BIN="$BIN" ARCH="$ARCH" -j2 > "$O/p2-make.log" 2>&1 || { tail -20 "$O/p2-make.log"; exit 3; }
  cp "$BIN"/*.ptxas "$O/" 2>/dev/null
  PY=$W/ref312/bin/python
  [ -x "$PY" ] || { uv venv -q "$W/ref312" --python 3.12 && uv pip install -q --python "$PY" numpy; } || exit 3
  "$PY" benchmarks/pous/bench.py --scheme "$SCHEME" --impl p2-gpu --p2-bin "$BIN" --scratch "$W/scratch" \
    --impl-commit "${RESEARCH_SOURCE_COMMIT:-}" --weight-set "$WS" --hardware "$HW" --out "$O/bench" --audits "$AUDITS" \
    ${MAX_SEGMENTS:+--max-segments "$MAX_SEGMENTS"}
else
  PY=$W/ref312/bin/python
  [ -x "$PY" ] || { uv venv -q "$W/ref312" --python 3.12 && uv pip install -q --python "$PY" numpy; } || exit 3
  "$PY" benchmarks/pous/bench.py --scheme "$SCHEME" --impl reference --impl-commit "${RESEARCH_SOURCE_COMMIT:-}" \
    --weight-set "$WS" --hardware "$HW" --out "$O/bench" --audits "$AUDITS" ${MAX_SEGMENTS:+--max-segments "$MAX_SEGMENTS"}
fi
