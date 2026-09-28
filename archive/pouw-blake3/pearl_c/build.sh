#!/usr/bin/env bash
# Pearl-C's shipped tree, built off the pod:
#   pearl_c.cubin    pearl_c.cu with CUDA 12.9.1's nvcc for sm_90a (-fmad=false, no -lineinfo, so the same bytes from any
#                    directory), refused unless ptxas reports no warning, spill or stack frame and the SASS has 4 QGMMA per
#                    128-deep block in every GEMM, 4 HGMMA in gemm_pearlc's peel, one QGMMA in each forming kernel,
#                    two codes per F2FP in form_s5's and quant_s5's e4m3 casts, and no .FTZ FADD, FFMA or FMUL outside
#                    the correctly rounded div, rcp and sqrt sequences ftz_pins.json pins for this nvcc (ftz_gate.py)
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
# every shipped cubin is built with these flags alone (no FTZ, no fast math); the .FTZ gate's pins are keyed to them
FLAGS="-O3 -std=c++17 -fmad=false"
"$C9/bin/nvcc" $FLAGS -gencode arch=compute_90a,code=sm_90a -Xptxas -v -cubin -I "$HERE" \
  -o "$OUT/pearl_c.cubin" "$HERE/pearl_c.cu" 2> "$OUT/ptxas.txt"
"$C9/bin/nvcc" $FLAGS -gencode arch=compute_120a,code=sm_120a -Xptxas -v -cubin -I "$HERE" \
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
json.dump({"verity/ml/kernels": funcs, "problems": problems}, open(f"{out}/sass.json", "w"), indent=1)
if problems:
    sys.exit("REFUSED: " + "; ".join(problems))
print(f"pearl_c.cubin: {len(funcs)} kernels; GEMMs " + ", ".join(f"{n} {funcs[n]['regs']} regs {funcs[n]['QGMMA']} QGMMA"
      for n in ("gemm_plain", "gemm_ct", "gemm_pearlc")) + f", peel {funcs['gemm_pearlc']['HGMMA']} HGMMA; no spills")
EOF
rm "$OUT/pearl_c.sass"
# the .FTZ gate (ftz_gate.py), keyed to this nvcc: no .FTZ FADD, FFMA or FMUL outside the correctly rounded div, rcp and sqrt
# sequences that ieee_ops and ieee_q pin; and the negative control, the same source under --use_fast_math, must fail it in them
NV=$("$C9/bin/nvcc" --version | grep -o 'V[0-9][0-9.]*' | head -1)
"$C9/bin/nvdisasm" "$OUT/pearl_c.cubin" > "$OUT/pearl_c.nvdisasm"
"$C9/bin/nvdisasm" "$OUT/hash_sm120a.cubin" > "$OUT/hash_sm120a.nvdisasm"
python3 "$HERE/ftz_gate.py" "$OUT/pearl_c.nvdisasm" "$NV/sm_90a $FLAGS" || { echo "REFUSED: the .FTZ gate (pearl_c.cubin)" >&2; exit 1; }
python3 "$HERE/ftz_gate.py" "$OUT/hash_sm120a.nvdisasm" "$NV/sm_120a $FLAGS" || { echo "REFUSED: the .FTZ gate (hash_sm120a.cubin)" >&2; exit 1; }
"$C9/bin/nvcc" $FLAGS --use_fast_math -gencode arch=compute_90a,code=sm_90a -cubin -I "$HERE" \
  -o "$OUT/fast_math.cubin" "$HERE/pearl_c.cu" 2> /dev/null
"$C9/bin/nvdisasm" "$OUT/fast_math.cubin" > "$OUT/fast_math.nvdisasm"
if python3 "$HERE/ftz_gate.py" "$OUT/fast_math.nvdisasm" "$NV/sm_90a $FLAGS" 2> "$OUT/fast_math.gate"; then
  echo "REFUSED: the .FTZ gate passed a --use_fast_math build" >&2; exit 1
fi
grep -q '^ieee_ops: \|^ieee_q: ' "$OUT/fast_math.gate" || { echo "REFUSED: the fast-math control did not fail in the test kernels" >&2; exit 1; }
rm "$OUT/pearl_c.nvdisasm" "$OUT/hash_sm120a.nvdisasm" "$OUT/fast_math.cubin" "$OUT/fast_math.nvdisasm" "$OUT/fast_math.gate"
echo ".FTZ gate: only nvcc $NV's pinned div, rcp and sqrt sequences; the --use_fast_math control fails in the test kernels"
(cd "$ROOT" && uv run python benchmarks/pouw/pearl_c/fixture.py "$OUT/check")
cp "$HERE/run.py" "$HERE/bench.py" "$HERE/bench.json" "$HERE/run.sh" "$HERE/vendor.py" "$OUT/"
{ "$C9/bin/nvcc" --version | tail -2; echo "flags: $FLAGS (sm_90a pearl_c.cubin, sm_120a hash_sm120a.cubin)"; } > "$OUT/toolchain.txt"
{ git -C "$ROOT" rev-parse HEAD; git -C "$ROOT" status --porcelain -- benchmarks/pouw/pearl_c verity/core/protocols/pouw verity; } > "$OUT/source.txt"
(cd "$OUT" && find . -type f ! -name MANIFEST.sha256 | LC_ALL=C sort | xargs sha256sum > MANIFEST.sha256)
du -sh "$OUT" | cut -f1 | xargs echo "tree:"
