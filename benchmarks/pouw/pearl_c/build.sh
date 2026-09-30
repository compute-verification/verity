#!/usr/bin/env bash
# Pearl-C's shipped tree, built off the pod:
#   pearl_c.cubin    pearl_c.cu with CUDA 12.9.1's nvcc for sm_90a (-fmad=false, no -lineinfo, so the same bytes from any
#                    directory), refused unless ptxas reports no warning, spill or stack frame and the SASS has 4 QGMMA per
#                    128-deep block in every GEMM, 4 HGMMA in gemm_pearlc's peel, one QGMMA in each forming kernel, and
#                    two codes per F2FP in form_s5's and quant_s5's e4m3 casts
#   hash_sm120a.cubin hash.cu (hash.cuh's BLAKE3 hashing kernels: -h1 messages, leaves and frame-b3 trees, tickets) for
#                    sm_120a, refused unless ptxas reports no warning, spill or stack frame
#   sass.json        registers and tensor-core, bulk-copy and local-memory instruction counts per kernel
#   check/           fixture.py's inputs and expected buffers (needs verity and numpy)
#   bench.json       the fixed-shape bench's shapes, keys and sampled tiles with the CPU twin's bytes (twin.py; committed)
#   run.py, bench.py, run.sh, vendor.py, toolchain.txt, source.txt (the verity commit), MANIFEST.sha256
# Usage: bash benchmarks/pouw/pearl_c/build.sh [OUT]   (default /tmp/pearl-c-ship; C9 names CUDA 12.9.1, default /tmp/cuda-12.9.1)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
C9=${C9:-/tmp/cuda-12.9.1}; OUT=${1:-/tmp/pearl-c-ship}
[ -x "$C9/bin/nvcc" ] || { echo "no nvcc at $C9 (CUDA 12.9.1)" >&2; exit 1; }
"$C9/bin/nvcc" --version | grep -q "release 12.9" || { echo "$C9 is not CUDA 12.9" >&2; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"
"$C9/bin/nvcc" -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -fmad=false -Xptxas -v -cubin -I "$HERE" \
  -o "$OUT/pearl_c.cubin" "$HERE/pearl_c.cu" 2> "$OUT/ptxas.txt"
"$C9/bin/nvcc" -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -fmad=false -Xptxas -v -cubin -I "$HERE" \
  -o "$OUT/hash_sm120a.cubin" "$HERE/hash.cu" 2> "$OUT/ptxas_sm120a.txt"
"$C9/bin/cuobjdump" -sass "$OUT/pearl_c.cubin" > "$OUT/pearl_c.sass"
python3 - "$OUT" <<'EOF'
import json, re, sys
out = sys.argv[1]
ptx = open(f"{out}/ptxas.txt").read()
problems = [l for l in ptx.splitlines() if re.search(r"warning|error|C7514|serializ", l, re.I)]
ptx120 = open(f"{out}/ptxas_sm120a.txt").read()
problems += [f"sm_120a: {l}" for l in ptx120.splitlines() if re.search(r"warning|error", l, re.I)]
problems += [f"sm_120a: {f}" for f, st, sp in re.findall(r"Compiling entry function '(\w+)' for 'sm_120a'.*?(\d+) bytes stack frame, "
                                                         r"(\d+) bytes spill stores", ptx120, re.S) if int(st) or int(sp)]
funcs = {}
for m in re.finditer(r"Compiling entry function '(\w+)' for 'sm_90a'(.*?)(?=Compiling entry function|\Z)", ptx, re.S):
    body = m.group(2)
    frame = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", body)
    funcs[m.group(1)] = {"regs": int(re.search(r"Used (\d+) registers", body).group(1)), "stack": int(frame.group(1)),
                         "spill_stores": int(frame.group(2)), "spill_loads": int(frame.group(3))}
cur, ops = None, ("QGMMA", "HGMMA", "UBLKCP", "SYNCS", "LDL", "STL", "F2FP")
for line in open(f"{out}/pearl_c.sass"):
    fm = re.search(r"Function : (\w+)", line)
    if fm:
        cur = fm.group(1)
        funcs.setdefault(cur, {}).update({op: 0 for op in ops} | {"F2FP_one_code": 0})
    elif cur:
        for op in ops:
            if re.search(r"\b" + op + r"\b", line):
                funcs[cur][op] += 1
        if re.search(r"\bF2FP\S*\s+R\d+,\s*RZ,", line):
            funcs[cur]["F2FP_one_code"] += 1
# the e4m3 casts of the forming and of the adopted baseline: two codes per F2FP (the statement's cast price), none alone
for name in ("form_s5", "quant_s5"):
    if not funcs[name]["F2FP"] or funcs[name]["F2FP_one_code"]:
        problems.append(f"{name}: {funcs[name]['F2FP_one_code']} of {funcs[name]['F2FP']} F2FP convert one code")
for name, r in funcs.items():
    if r.get("stack") or r.get("spill_stores") or r.get("spill_loads") or r["LDL"] or r["STL"]:
        problems.append(f"{name}: local memory {r}")
for name, (q, h) in {"gemm_plain": (4, 0), "gemm_fast": (4, 0), "gemm_ct": (4, 0), "gemm_pearlc": (4, 4), "gemm_words": (4, 0),
                     "form": (1, 0), "form_s5": (1, 0)}.items():
    r = funcs[name]
    if r["QGMMA"] < q or r["QGMMA"] % q or r["HGMMA"] != h:
        problems.append(f"{name}: {r['QGMMA']} QGMMA, {r['HGMMA']} HGMMA (want a multiple of {q}, and {h})")
json.dump({"kernels": funcs, "problems": problems}, open(f"{out}/sass.json", "w"), indent=1)
if problems:
    sys.exit("REFUSED: " + "; ".join(problems))
print(f"pearl_c.cubin: {len(funcs)} kernels; GEMMs " + ", ".join(f"{n} {funcs[n]['regs']} regs {funcs[n]['QGMMA']} QGMMA"
      for n in ("gemm_plain", "gemm_ct", "gemm_pearlc")) + f", peel {funcs['gemm_pearlc']['HGMMA']} HGMMA; no spills")
EOF
rm "$OUT/pearl_c.sass"
(cd "$ROOT" && uv run python benchmarks/pouw/pearl_c/fixture.py "$OUT/check")
cp "$HERE/run.py" "$HERE/bench.py" "$HERE/bench.json" "$HERE/run.sh" "$HERE/vendor.py" "$OUT/"
"$C9/bin/nvcc" --version | tail -2 > "$OUT/toolchain.txt"
{ git -C "$ROOT" rev-parse HEAD; git -C "$ROOT" status --porcelain -- benchmarks/pouw/pearl_c protocols/pouw packages/verity; } > "$OUT/source.txt"
(cd "$OUT" && find . -type f ! -name MANIFEST.sha256 | LC_ALL=C sort | xargs sha256sum > MANIFEST.sha256)
du -sh "$OUT" | cut -f1 | xargs echo "tree:"
