# Sourced before build.sh on a machine without CUDA 12.9 (the RTX PRO 6000 node has 13.0), so every build uses the one
# nvcc the baselines were measured with:
#
#   . benchmarks/pouw/harness/jobs/toolkit.sh && benchmarks/pouw/harness/native/build.sh
#
# Assembles, once and under a lock, CUDA 12.9.1's nvcc, cudart, cccl, cuobjdump and nvdisasm from NVIDIA's redistributable
# archives (each pinned by sha256) and the cuBLASLt headers and library from the wheel env.sh pins, into POUW_HARNESS_TOOLKIT
# (default /workspace/cache/pouw-cuda-12.9.1), and exports CUDA_HOME. Needs curl, tar and xz, and uv or pip.

_tk_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POUW_HARNESS_TOOLKIT="${POUW_HARNESS_TOOLKIT:-/workspace/cache/pouw-cuda-12.9.1}"
_tk_url="https://developer.download.nvidia.com/compute/cuda/redist"
_tk_parts="cuda_nvcc 12.9.86 7a1a5b652e5ef85c82b721d10672fc9a2dbaab44e9bd3c65a69517bf53998c35
cuda_cudart 12.9.79 1f6ad42d4f530b24bfa35894ccf6b7209d2354f59101fd62ec4a6192a184ce99
cuda_cccl 12.9.27 8b1a5095669e94f2f9afd7715533314d418179e9452be61e2fde4c82a3e542aa
cuda_cuobjdump 12.9.82 ee0de40e8c18068bfcc53e73510e7e7a1a80555205347940df67fa525d24452f
cuda_nvdisasm 12.9.88 49296dd550e05434185a8588ec639f1325b2de413e2321ddd7e56c5182a476ff"
_tk_cublas="$(grep -o 'nvidia-cublas-cu12==[0-9.]*' "$_tk_here/env.sh")"
_tk_want="$(printf '%s\n%s\n' "$_tk_parts" "$_tk_cublas" | sha256sum | cut -c1-16)"
mkdir -p "$POUW_HARNESS_TOOLKIT" || return 1
(
  set -euo pipefail
  flock 9
  t="$POUW_HARNESS_TOOLKIT"
  [[ "$(cat "$t/ready" 2>/dev/null)" == "$_tk_want" ]] && exit 0
  rm -rf "$t/root" "$t/dl" "$t/wheel"
  mkdir -p "$t/root/targets/x86_64-linux" "$t/dl"
  while read -r name ver sha; do
    f="$t/dl/$name-linux-x86_64-$ver-archive.tar.xz"
    curl -fsSL --retry 4 -o "$f" "$_tk_url/$name/linux-x86_64/$(basename "$f")"
    echo "$sha  $f" | sha256sum -c --quiet
    tar -xJf "$f" -C "$t/root" --strip-components=1
  done <<<"$_tk_parts"
  if command -v uv >/dev/null; then
    uv pip install -q --no-deps --target "$t/wheel" "$_tk_cublas"
  else
    python3 -m pip install -q --no-deps --target "$t/wheel" "$_tk_cublas"
  fi
  cp -a "$t/wheel/nvidia/cublas/include/." "$t/root/include/"
  cp -a "$t/wheel/nvidia/cublas/lib/." "$t/root/lib/"
  for so in libcublas libcublasLt; do ln -sf "$so.so.12" "$t/root/lib/$so.so"; done
  ln -sfn lib "$t/root/lib64"
  ln -sfn ../../include "$t/root/targets/x86_64-linux/include"
  ln -sfn ../../lib "$t/root/targets/x86_64-linux/lib"
  rm -rf "$t/dl" "$t/wheel"
  "$t/root/bin/nvcc" --version | tail -1
  echo "$_tk_want" > "$t/ready"
) 9>"$POUW_HARNESS_TOOLKIT/.lock" || return 1
export CUDA_HOME="$POUW_HARNESS_TOOLKIT/root"
