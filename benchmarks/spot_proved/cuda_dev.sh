#!/usr/bin/env bash
# The CUDA headers FlashInfer's JIT reads (cuBLASLt, cuRAND, NVRTC) for the toolkit nvcc resolves to, from NVIDIA's apt repo.
set -euo pipefail
NVCC=$(command -v nvcc || ls -d /usr/local/cuda/bin/nvcc 2>/dev/null || true)
HOME_CUDA=${CUDA_HOME:-$(dirname "$(dirname "$(readlink -f "${NVCC:-/usr/local/cuda/bin/nvcc}")")")}
VER=$(basename "$HOME_CUDA" | sed -n 's/^cuda-\([0-9]*\)\.\([0-9]*\)$/\1-\2/p')
echo "nvcc=$NVCC cuda_home=$HOME_CUDA ver=${VER:-?}"
ls -d /usr/local/cuda* || true
missing=0
for h in cublasLt.h curand_kernel.h nvrtc.h; do [ -f "$HOME_CUDA/include/$h" ] || { echo "missing $h"; missing=1; }; done
[ "$missing" = 0 ] && { echo "headers present"; exit 0; }
[ -n "$VER" ] || { echo "cannot tell the toolkit version from $HOME_CUDA"; exit 2; }
if ! apt-cache policy "libcublas-dev-$VER" 2>/dev/null | grep -q Candidate; then
  apt-get update -qq >/dev/null 2>&1 || true
fi
if ! apt-cache policy "libcublas-dev-$VER" 2>/dev/null | grep -q "Candidate: [0-9]"; then
  . /etc/os-release
  curl -fsSLo /tmp/cuda-keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID//./}/x86_64/cuda-keyring_1.1-1_all.deb"
  dpkg -i /tmp/cuda-keyring.deb >/dev/null
  apt-get update -qq >/dev/null
fi
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "libcublas-dev-$VER" "libcurand-dev-$VER" "cuda-nvrtc-dev-$VER" \
  "cuda-cudart-dev-$VER" >/dev/null
for h in cublasLt.h curand_kernel.h nvrtc.h; do ls -l "$HOME_CUDA/include/$h"; done
