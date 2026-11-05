#!/usr/bin/env bash
# The decode-to-staging ship for one GPU arch, built off the card on a CPU node (cwd = the source tree's root):
#   lib/<variant>/libp2stage.so  p2stage.cu for ARCH (default sm_90a, the H200's), each refused unless ptxas reports no warning,
#                                spill or stack frame and its SASS has no local memory (sass_gate.py, which records registers,
#                                shared memory and the instruction mix into lib/<variant>/sass.json); VARIANTS names them
#   p2v1/libp2v1.so              memory-accounting's benchmarks/pous/p2_v1 library by its own Makefile (FAST's default: v4 keys,
#                                Karatsuba squaring), CGBN at p2_gpu/CGBN.commit: decode_bench.py's rates, the encode rate, the
#                                codewords and the staging gate's second decoder
#   toolchain.txt (nvcc --version and every command line), source.txt (the commit), MANIFEST.sha256
# Usage: bash benchmarks/torch_gpu/s8_dec_linear/build.sh OUT. CUDA names a toolkit (nvcc >= 12.8), else the node's
# /usr/local/cuda-13.0 or /usr/local/cuda, else NVIDIA's redistributable archives of CUDA_REDIST (default 13.0.2: the H200's
# toolkit is 13.0), each held to its published sha256, unpacked into a scratch directory that is deleted.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
OUT=${1:?usage: build.sh OUT}; ARCH=${ARCH:-sm_90a}
VARIANTS=${VARIANTS:-"w2:-DP2ST_WARPS=2 w4:-DP2ST_WARPS=4 w2sb:-DP2ST_WARPS=2:-DP2ST_SCHOOLBOOK"}
S=$(mktemp -d "${TMPDIR:-/tmp}/s8-dec-build.XXXXXX"); trap 'rm -rf "$S"' EXIT
rm -rf "$OUT"; mkdir -p "$OUT/lib" "$OUT/p2v1"
: > "$OUT/toolchain.txt"

release_ok() { "$1/bin/nvcc" --version 2>/dev/null | grep -Eq "release 12\.(8|9)|release 1[3-9]\."; }
fetch_redist() {   # $1 version, $2 destination: the components nvcc needs to compile and link a shared library and dump SASS
  python3 - "$1" "$2" <<'EOF'
import hashlib, json, subprocess, sys, urllib.request
from pathlib import Path
ver, dest = sys.argv[1], Path(sys.argv[2])
base = "https://developer.download.nvidia.com/compute/cuda/redist/"
doc = json.load(urllib.request.urlopen(base + f"redistrib_{ver}.json", timeout=60))
dest.mkdir(parents=True, exist_ok=True)
for name in ("cuda_nvcc", "cuda_crt", "libnvvm", "cuda_cudart", "cuda_cccl", "cuda_cuobjdump", "cuda_nvdisasm"):
    if name not in doc:
        continue
    e = doc[name]["linux-x86_64"]
    data = urllib.request.urlopen(base + e["relative_path"], timeout=600).read()
    if hashlib.sha256(data).hexdigest() != e["sha256"]:
        sys.exit(f"{name}: sha256 differs from redistrib_{ver}.json's")
    subprocess.run(["tar", "-xJ", "-C", str(dest), "--strip-components=1"], input=data, check=True)
    print(f"{name} {e['relative_path']} {e['sha256']}")
EOF
  ln -sfn lib "$2/lib64"
  mkdir -p "$2/targets"; ln -sfn .. "$2/targets/x86_64-linux"
}
CUDA=${CUDA:-}
if [ -z "$CUDA" ]; then
  for home in /usr/local/cuda-13.0 /usr/local/cuda; do release_ok "$home" && { CUDA=$home; break; }; done
fi
if [ -z "$CUDA" ]; then
  CUDA=$S/cuda; fetch_redist "${CUDA_REDIST:-13.0.2}" "$CUDA" >> "$OUT/toolchain.txt"
fi
release_ok "$CUDA" || { echo "no nvcc >= 12.8 at $CUDA" >&2; exit 1; }
NVCC=$CUDA/bin/nvcc
{ echo "CUDA=$CUDA"; "$NVCC" --version; } >> "$OUT/toolchain.txt"

for v in $VARIANTS; do
  name=${v%%:*}; IFS=: read -r -a defs <<< "${v#*:}"
  d=$OUT/lib/$name; mkdir -p "$d"
  cmd=("$NVCC" -O3 -std=c++17 -gencode "arch=compute_${ARCH#sm_},code=$ARCH" -Xcompiler -fPIC -shared "${defs[@]}"
       -I "$CUDA/include" -L "$CUDA/lib" -Xptxas -v -o "$d/libp2stage.so" "$HERE/p2stage.cu")
  echo "${cmd[*]}" >> "$OUT/toolchain.txt"
  "${cmd[@]}" 2> "$d/ptxas.txt" || { cat "$d/ptxas.txt" >&2; exit 1; }
  PATH="$CUDA/bin:$PATH" "$CUDA/bin/cuobjdump" -sass "$d/libp2stage.so" > "$S/$name.sass"
  python3 "$HERE/sass_gate.py" "$ARCH" "$d/ptxas.txt" "$S/$name.sass" "$d/sass.json" | sed "s/^/$name: /"
done

# memory-accounting's library, unmodified: its Makefile's arch flag takes sm_90 (no wgmma), its host pass gmp.h
GMPINC=
if ! echo '#include <gmp.h>' | cpp >/dev/null 2>&1; then
  uv pip install -q --target "$S/py" gmpy2
  GMPINC=$(find "$S/py" -name gmp.h -printf '%h\n' | head -1)
  [ -n "$GMPINC" ] || { echo "gmpy2's wheel has no gmp.h" >&2; exit 1; }
fi
git clone -q https://github.com/NVlabs/CGBN "$S/CGBN"
git -C "$S/CGBN" checkout -q "$(cat "$ROOT/benchmarks/pous/p2_gpu/CGBN.commit")"
P2ARCH=${ARCH%a}
mk=(make -C "$ROOT/benchmarks/pous/p2_v1" NVCC="$NVCC -I $CUDA/include -L $CUDA/lib" CGBN="$S/CGBN" BIN="$OUT/p2v1" ARCH="$P2ARCH" ${GMPINC:+"INC=-I $GMPINC"})
echo "${mk[*]}" >> "$OUT/toolchain.txt"
"${mk[@]}" > "$S/p2v1-make.log" 2>&1 || { tail -30 "$S/p2v1-make.log" >&2; exit 1; }
grep -E "Compiling entry|registers|spill" "$OUT/p2v1/libp2v1.so.ptxas" | head -40 > "$OUT/p2v1/ptxas-summary.txt" || true

{ git -C "$ROOT" rev-parse HEAD; git -C "$ROOT" status --porcelain -- benchmarks/torch_gpu/s8_dec_linear benchmarks/pous/p2_v1 benchmarks/pous/p2_gpu; } > "$OUT/source.txt"
(cd "$OUT" && find . -type f ! -name MANIFEST.sha256 | LC_ALL=C sort | xargs sha256sum > MANIFEST.sha256)
du -sh "$OUT" | cut -f1 | xargs echo "ship:"
