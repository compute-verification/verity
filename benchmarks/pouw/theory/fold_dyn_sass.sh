#!/usr/bin/env bash
# FOLD-DYN's honest steps on sm_90a: build fold_dyn_sass.cu, dump its SASS and count each kernel's instructions a step
# and output (64 steps, 8 outputs a thread). Needs nvcc and cuobjdump (a CPU node with the toolkit); writes
# fold_dyn_sass.sass and fold_dyn_sass.json under $OUT (default .).
#   usage: bash fold_dyn_sass.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-.}
NVCC=${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
CUOBJDUMP=${CUOBJDUMP:-$(dirname "$NVCC")/cuobjdump}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
"$NVCC" --version | tail -n 1
"$NVCC" -arch=sm_90a -O3 -cubin -o "$WORK/f.cubin" "$HERE/fold_dyn_sass.cu"
"$CUOBJDUMP" -sass "$WORK/f.cubin" >"$OUT/fold_dyn_sass.sass"
python3 - "$OUT/fold_dyn_sass.sass" "$OUT/fold_dyn_sass.json" <<'EOF'
import collections, json, re, sys
kern, counts = None, collections.defaultdict(collections.Counter)
for line in open(sys.argv[1]):
    m = re.search(r"Function : (\S+)", line)
    if m:
        kern = m.group(1)
        continue
    m = re.search(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)", line)
    if kern and m:
        counts[kern][m.group(1)] += 1
per = 64 * 8
out = {k: {op: n / per for op, n in sorted(c.items()) if n >= per // 4} for k, c in counts.items()}
json.dump({"per": "lane instructions a step and output (64 steps, 8 outputs a thread)", "kernels": out}, open(sys.argv[2], "w"), indent=1)
for k, c in out.items():
    print(k, " ".join(f"{op}={v:.3f}" for op, v in c.items()))
EOF
