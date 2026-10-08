#!/usr/bin/env bash
# h1_bench.py's ship, built off the node:
#   hash_sm120a.cubin  hash.cu (hash.cuh and hash_sm120.cuh) with CUDA 12.9.1 for sm_120a, build.sh's flags: the -h1 kernels as pushed and the tuned ones
#                      (hash_rows_w, hash_tree_small, hash_leaf_p, hash_msg_v), refused on a ptxas warning, spill or stack frame
#   h1_standin.cubin   h1_standin.cu: the decode GEMM's weight traffic (not a GEMM), derive_rows and the read-only L2 flush
#   h1_vectors.json    the device gates, from the verifier's reference (h1_vectors.py; needs verity and blake3)
#   h1_bench.py, run.py, MANIFEST.sha256
# Usage: bash benchmarks/pouw/pearl_c/h1_ship.sh [OUT]   (default /tmp/h1-ship; C9 names CUDA 12.9.1, default /tmp/cuda-12.9.1)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
C9=${C9:-/tmp/cuda-12.9.1}; OUT=${1:-/tmp/h1-ship}
"$C9/bin/nvcc" --version | grep -q "release 12.9" || { echo "$C9 is not CUDA 12.9" >&2; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"
"$C9/bin/nvcc" -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -fmad=false -Xptxas -v -cubin -I "$HERE" \
  -o "$OUT/hash_sm120a.cubin" "$HERE/hash.cu" 2> "$OUT/ptxas_sm120a.txt"
"$C9/bin/nvcc" -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -Xptxas -v -cubin \
  -o "$OUT/h1_standin.cubin" "$HERE/h1_standin.cu" 2> "$OUT/ptxas_standin.txt"
python3 - "$OUT" <<'EOF'
import re, sys
out = sys.argv[1]
text = open(f"{out}/ptxas_sm120a.txt").read() + open(f"{out}/ptxas_standin.txt").read()
bad = [l for l in text.splitlines() if re.search(r"warning|error", l, re.I)]
bad += [f for f, st, sp in re.findall(r"Compiling entry function '(\w+)'.*?(\d+) bytes stack frame, (\d+) bytes spill stores", text, re.S)
        if int(st) or int(sp)]
if bad:
    sys.exit(f"refused: {bad}")
EOF
(cd "$ROOT" && uv run python benchmarks/pouw/pearl_c/h1_vectors.py "$OUT/h1_vectors.json")
cp "$HERE/h1_bench.py" "$HERE/run.py" "$OUT/"
(cd "$OUT" && sha256sum hash_sm120a.cubin h1_standin.cubin h1_vectors.json h1_bench.py run.py > MANIFEST.sha256)
tar -cf "$OUT.tar" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo "$OUT.tar"
