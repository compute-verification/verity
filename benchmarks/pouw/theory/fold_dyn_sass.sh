#!/usr/bin/env bash
# FOLD-DYN's honest steps: build fold_dyn_sass.cu for each of $ARCHS (default sm_90a and sm_120), dump its SASS and
# count each kernel's instructions a step and output (64 steps, 8 outputs a thread). Needs nvcc and cuobjdump: a node
# with the toolkit, or CUDA's redistributable archives (cuda_nvcc, cuda_cuobjdump, cuda_cudart, cuda_cccl) unpacked
# into one prefix whose bin/ is on PATH. Writes fold_dyn_sass.<arch>.sass and fold_dyn_sass.json under $OUT (default .).
#   usage: bash fold_dyn_sass.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-.}
ARCHS=${ARCHS:-sm_90a sm_120}
NVCC=${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
CUOBJDUMP=${CUOBJDUMP:-$(dirname "$NVCC")/cuobjdump}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
"$NVCC" --version | tail -n 1
for a in $ARCHS; do
    "$NVCC" -arch="$a" -O3 -cubin -o "$WORK/$a.cubin" "$HERE/fold_dyn_sass.cu"
    "$CUOBJDUMP" -sass "$WORK/$a.cubin" >"$OUT/fold_dyn_sass.$a.sass"
done
python3 - "$OUT" $ARCHS <<'EOF'
import collections, json, re, sys
out_dir, archs = sys.argv[1], sys.argv[2:]
per = 64 * 8
res = {}
for a in archs:
    kern, counts = None, collections.defaultdict(collections.Counter)
    for line in open(f"{out_dir}/fold_dyn_sass.{a}.sass"):
        m = re.search(r"Function : (\S+)", line)
        if m:
            kern = m.group(1)
            continue
        m = re.search(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)", line)
        if kern and m:
            counts[kern][m.group(1)] += 1
    res[a] = {k: {op: n / per for op, n in sorted(c.items()) if n >= per // 4} for k, c in counts.items()}
    for k, c in res[a].items():
        print(a, k, " ".join(f"{op}={v:.3f}" for op, v in c.items()), f"total={sum(c.values()):.3f}")
json.dump({"per": "lane instructions a step and output (64 steps, 8 outputs a thread)", "archs": res},
          open(f"{out_dir}/fold_dyn_sass.json", "w"), indent=1)
EOF
