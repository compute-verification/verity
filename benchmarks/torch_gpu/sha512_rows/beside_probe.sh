#!/usr/bin/env bash
# beside_probe.cu's kernels, each its own nvcc run for sm_90a, on a CPU node with CUDA 13: ptxas's verdict, registers and
# spills a kernel, into $OUT/probe-N.log, and the SASS's setmaxnreg and wgmma lines.
set -uo pipefail
O=${OUT:?}
CUDA=${CUDA:-/usr/local/cuda-13.0}
X=benchmarks/torch_gpu/sha512_rows
"$CUDA/bin/nvcc" --version | tail -2 > "$O/nvcc.txt"
for n in 1 2 3 4 5; do
  "$CUDA/bin/nvcc" -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -Xptxas -v -cubin -DPROBE=$n \
    -o "$O/probe-$n.cubin" $X/beside_probe.cu > "$O/probe-$n.log" 2>&1
  echo "probe $n rc $?" | tee -a "$O/rc.txt"
  grep -E "error|warning|registers|spill|Compiling" "$O/probe-$n.log" | sed "s/^/  /"
  [ -s "$O/probe-$n.cubin" ] && "$CUDA/bin/cuobjdump" -sass "$O/probe-$n.cubin" | grep -E "USETMAXREG|WARPGROUP|HGMMA|IGMMA" |
    sort | uniq -c | sed "s/^/  /"
done
