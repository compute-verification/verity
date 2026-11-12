#!/usr/bin/env bash
# emp.sh build at the H100 price rounds' toolkit pin, nvcc and ptxas 12.4.131, taken from NVIDIA's PyPI wheels, so an
# H200 round built here differs from those rounds in the card and not in the compiler.
#
#   research run --on auto --cwd source -- bash benchmarks/pouw/dc/emp_cuda124.sh SRC...
#
# The binaries link cudart 12.4 shared; libcudart.so.12 goes into bins.tgz beside them (rpath $ORIGIN, mode 644 so
# emp.sh time skips it), and gating loads it from the wheel. $OUT/ptxas.txt is the ptxas that built them. SASS is read
# by the node's own cuobjdump, which disassembles a 12.4 cubin as it is.
set -euo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NVCC_V=12.4.131
CUDART_V=12.4.127
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
uv pip install -q --target "$T/py" "nvidia-cuda-nvcc-cu12==$NVCC_V" "nvidia-cuda-runtime-cu12==$CUDART_V"
N=$T/py/nvidia/cuda_nvcc
R=$T/py/nvidia/cuda_runtime
mkdir -p "$T/bin" "$T/lib"
ln -s "$R/lib/libcudart.so.12" "$T/lib/libcudart.so"
extra=()
gcc_major=$(gcc -dumpversion | cut -d. -f1)
[ "$gcc_major" -le 13 ] || extra=(-allow-unsupported-compiler)
cat >"$T/bin/nvcc" <<EOF
#!/usr/bin/env bash
exec "$N/bin/nvcc" ${extra[*]} -I"$R/include" -L"$T/lib" -cudart shared -Xlinker -rpath,'\$ORIGIN' "\$@"
EOF
chmod +x "$T/bin/nvcc"
{ "$N/bin/ptxas" --version; echo "host gcc $(gcc -dumpversion)${extra:+ (${extra[*]})}"; } >"$OUT/ptxas.txt"
export LD_LIBRARY_PATH="$R/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
rc=0
NVCC="$T/bin/nvcc" bash "$HERE/emp.sh" build "$@" || rc=$?
if [ -f "$OUT/bins.tgz" ]; then
  mkdir "$T/bins"
  tar -C "$T/bins" -xzf "$OUT/bins.tgz"
  install -m 644 "$R/lib/libcudart.so.12" "$T/bins/libcudart.so.12"
  tar -C "$T/bins" -czf "$OUT/bins.tgz" .
fi
exit $rc
