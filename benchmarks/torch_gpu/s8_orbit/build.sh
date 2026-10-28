#!/usr/bin/env bash
# s8_orbit.cubin for sm_90a (the H100 and H200), built off the card, and the ship run.sh unpacks there: the cubin, ptxas's
# report (refused on a spill, local memory or a serialized wgmma), its SASS's instruction counts per kernel (sass.json),
# the toolchain, the commit and MANIFEST.sha256.  On a CPU node with CUDA >= 12.8 (CUDA names the toolkit, default
# /usr/local/cuda-13.0):  bash benchmarks/torch_gpu/s8_orbit/build.sh OUT
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
CUDA=${CUDA:-/usr/local/cuda-13.0}; OUT=${1:?usage: build.sh OUT}
[ -x "$CUDA/bin/nvcc" ] || { echo "no nvcc at $CUDA" >&2; exit 1; }
"$CUDA/bin/nvcc" --version | grep -Eq "release 12\.(8|9)|release 1[3-9]\." || { echo "$CUDA's nvcc is older than 12.8" >&2; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"
FLAGS=(-O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -fmad=false -Xptxas -v -cubin)
"$CUDA/bin/nvcc" "${FLAGS[@]}" -o "$OUT/s8_orbit.cubin" "$HERE/s8_orbit.cu" 2> "$OUT/ptxas.txt" || { cat "$OUT/ptxas.txt" >&2; exit 1; }
cat "$OUT/ptxas.txt"
"$CUDA/bin/cuobjdump" -sass "$OUT/s8_orbit.cubin" > "$OUT/s8_orbit.sass"
python3 - "$OUT" <<'EOF'
import json, re, sys
from collections import Counter
out = sys.argv[1]
ptx = open(f"{out}/ptxas.txt").read()
problems = []
if re.search(r"serializ", ptx):
    problems.append("ptxas serialized a wgmma")
for m in re.finditer(r"(\d+) bytes spill stores, (\d+) bytes spill loads", ptx):
    if int(m.group(1)) or int(m.group(2)):
        problems.append(f"spills: {m.group(0)}")
for m in re.finditer(r"Used (\d+) registers, used \d+ barriers(?:, (\d+) bytes lmem)?", ptx):
    if m.group(2) and int(m.group(2)):
        problems.append(f"local memory: {m.group(0)}")
kernels, cur = {}, None
for line in open(f"{out}/s8_orbit.sass"):
    k = re.match(r"\s+Function : (\S+)", line)
    if k:
        cur = kernels.setdefault(k.group(1), Counter())
        continue
    op = re.match(r"\s+/\*[0-9a-f]{4}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)", line)
    if cur is not None and op:
        cur[op.group(1)] += 1
for name, c in kernels.items():
    if name.startswith("s8orbit_") and name != "s8orbit_layout":
        if not any(o.startswith("HGMMA") or o.startswith("IGMMA") for o in c):
            problems.append(f"{name}: no IGMMA")
        if c.get("STL", 0) or c.get("LDL", 0):
            problems.append(f"{name}: local memory in SASS")
regs = {m.group(1): int(m.group(2)) for m in re.finditer(r"Function properties for (\S+)[\s\S]*?Used (\d+) registers", ptx)}
doc = {"problems": problems, "registers": regs,
       "kernels": {n: {"igmma": {o: v for o, v in c.items() if "GMMA" in o}, "instructions": sum(c.values())} for n, c in kernels.items()}}
json.dump(doc, open(f"{out}/sass.json", "w"), indent=1)
print(json.dumps(doc, indent=1))
sys.exit(1 if problems else 0)
EOF
{ "$CUDA/bin/nvcc" --version; echo "nvcc ${FLAGS[*]} -o s8_orbit.cubin s8_orbit.cu"; } > "$OUT/toolchain.txt"
{ git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}"
  git -C "$ROOT" status --porcelain -- benchmarks/torch_gpu/s8_orbit 2>/dev/null || true; } > "$OUT/source.txt"
(cd "$OUT" && find . -type f ! -name MANIFEST.sha256 | LC_ALL=C sort | xargs sha256sum > MANIFEST.sha256)
