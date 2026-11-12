#!/usr/bin/env bash
# emp.sh build with one CUDA release's nvcc and ptxas, taken from NVIDIA's redistributable archives (each checked
# against the release manifest's sha256), so a build's toolkit is pinned by its release and not by the node it runs on.
#
#   research run --on auto --cwd source -- bash benchmarks/pouw/dc/emp_cuda.sh RELEASE SRC...
#
# RELEASE 12.4.1 is nvcc and ptxas 12.4.131, the H100 price rounds' pin; 13.0.1 is 13.0.88. $OUT/toolkit.txt holds
# the release, each component's version and sha256, nvcc's and ptxas's --version and the host compiler.
set -euo pipefail
OUT=${OUT:?run under research run}
REL=${1:?usage: emp_cuda.sh RELEASE SRC...}
shift
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
URL=https://developer.download.nvidia.com/compute/cuda/redist
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
curl -fsS "$URL/redistrib_$REL.json" -o "$T/manifest.json"
python3 - "$T/manifest.json" >"$T/parts.tsv" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
for name in ("cuda_nvcc", "cuda_crt", "libnvvm", "cuda_cudart", "cuda_cccl", "cuda_cuobjdump", "cuda_nvdisasm"):
    if name in m:
        p = m[name]["linux-x86_64"]
        print(name, m[name]["version"], p["relative_path"], p["sha256"], sep="\t")
EOF
C=$T/cuda
mkdir -p "$C/targets"
echo "CUDA $REL ($URL/redistrib_$REL.json)" >"$OUT/toolkit.txt"
while IFS=$'\t' read -r name ver path sha; do
  curl -fsS "$URL/$path" -o "$T/part.tar.xz"
  echo "$sha  $T/part.tar.xz" | sha256sum -c --quiet
  tar -C "$C" --strip-components=1 -xJf "$T/part.tar.xz"
  rm "$T/part.tar.xz"
  echo "$name $ver sha256:$sha" >>"$OUT/toolkit.txt"
done <"$T/parts.tsv"
ln -s .. "$C/targets/x86_64-linux"
[ -e "$C/lib64" ] || ln -s lib "$C/lib64"
gcc_major=$(gcc -dumpversion | cut -d. -f1)
nvcc_major=$("$C/bin/nvcc" --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
# nvcc 12.4 takes host GCC up to 13; a newer one changes host code only, not the SASS the probes time.
[ "$nvcc_major" != 12 ] || [ "$gcc_major" -le 13 ] || export NVCC_APPEND_FLAGS=-allow-unsupported-compiler
{ "$C/bin/nvcc" --version; "$C/bin/ptxas" --version; echo "host gcc $(gcc -dumpversion) ${NVCC_APPEND_FLAGS:-}"; } \
  >>"$OUT/toolkit.txt"
NVCC="$C/bin/nvcc" bash "$HERE/emp.sh" build "$@"
