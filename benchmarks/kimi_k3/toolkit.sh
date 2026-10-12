# Sourced by the Kimi-K3 jobs on a machine without CUDA 12.9 (vy-nebius-cpu-1 has 13.0 only):
#
#   . benchmarks/kimi_k3/toolkit.sh
#
# Assembles, once and under a lock, the CUDA 12.9.1 toolkit a vLLM build at d9105ea80 needs (nvcc, the runtime and math
# libraries torch's CMake config looks for, cuobjdump and nvdisasm) from NVIDIA's redistributable archives into
# K3_TOOLKIT (default /workspace/research/cache/kimi-k3-prep/cuda-12.9.1), and exports CUDA_HOME and PATH. Every archive is
# checked against the sha256 in redistrib_12.9.1.json, whose own sha256 is pinned here. Needs curl, tar, xz and python3.

K3_TOOLKIT="${K3_TOOLKIT:-/workspace/research/cache/kimi-k3-prep/cuda-12.9.1}"
_k3_url="https://developer.download.nvidia.com/compute/cuda/redist"
_k3_manifest_sha="8335301010b0023ee1ff61eb11e2600ca62002d76780de4089011ad77e0c7630"
_k3_parts="cuda_nvcc cuda_cudart cuda_cccl cuda_nvrtc cuda_nvtx cuda_cupti cuda_profiler_api cuda_nvml_dev cuda_cuobjdump
cuda_nvdisasm cuda_nvprune libcublas libcusparse libcusolver libcurand libcufft libnvjitlink"
_k3_want="$(printf '%s\n%s\n' "$_k3_manifest_sha" "$_k3_parts" | sha256sum | cut -c1-16)"
mkdir -p "$K3_TOOLKIT" || return 1
(
  set -euo pipefail
  flock 9
  t="$K3_TOOLKIT"
  [[ "$(cat "$t/ready" 2>/dev/null)" == "$_k3_want" ]] && exit 0
  rm -rf "$t/root" "$t/dl"
  mkdir -p "$t/root" "$t/dl"
  m="$t/dl/redistrib_12.9.1.json"
  curl -fsSL --retry 4 -o "$m" "$_k3_url/redistrib_12.9.1.json"
  echo "$_k3_manifest_sha  $m" | sha256sum -c --quiet
  for name in $_k3_parts; do
    read -r rel sha < <(python3 "$(dirname "${BASH_SOURCE[0]}")/redist_entry.py" "$m" "$name")
    f="$t/dl/$(basename "$rel")"
    curl -fsSL --retry 4 -o "$f" "$_k3_url/$rel"
    echo "$sha  $f" | sha256sum -c --quiet
    tar -xJf "$f" -C "$t/root" --strip-components=1
    rm -f "$f"
  done
  ln -sfn lib "$t/root/lib64"
  mkdir -p "$t/root/targets/x86_64-linux"
  ln -sfn ../../include "$t/root/targets/x86_64-linux/include"
  ln -sfn ../../lib "$t/root/targets/x86_64-linux/lib"
  "$t/root/bin/nvcc" --version | tail -1
  echo "$_k3_want" > "$t/ready"
) 9>"$K3_TOOLKIT/.lock" || return 1
export CUDA_HOME="$K3_TOOLKIT/root"
export PATH="$CUDA_HOME/bin:$PATH"
