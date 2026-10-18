#!/usr/bin/env bash
# pouw_decode on one RTX PRO 6000, everything kept under $OUT:
#   off the lease  Pearl-C's cubin (pearl_c_sm120.cu, build.sh's flags) and pd.cubin, with ptxas's reports and their sha256s;
#                  host_test.py (pd's launches against Pearl-C's and the Python references on the CPU)
#   one GPU        bench.py: the check, then the interleaved timing (bench.jsonl); then bench.py --trace under nsys
#                  (--cuda-graph-trace=node)
#   off the lease  the trace to SQLite and trace.py over it (trace.json, trace.txt)
# Launch:
#   research run --on vy-nebius-2 --project verity --campaign hash-loop --source . --cwd source \
#     --env GPU_LEASE_WHO=kernels-pouw-decode -- bash benchmarks/hash/pouw_decode/run.sh
# REPS (default 30), ROWS (default 32,128), SHAPES (default qkv,o,gate_up,down), PD_SIDE (the variants of the schedules with
# side lanes, pd_host.SIDE, default 3,19,23) pass through.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
CUDA=${CUDA:-/usr/local/cuda}
: "${OUT:?run under research run, which sets OUT}"
REPS=${REPS:-30}; ROWS=${ROWS:-32,128}; SHAPES=${SHAPES:-qkv,o,gate_up,down}
SCR=$(mktemp -d); trap 'rm -rf "$SCR"' EXIT

FLAGS=(-O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -fmad=false -Xptxas -v -cubin)
{ "$CUDA/bin/nvcc" --version; echo "nvcc ${FLAGS[*]}"; g++ --version | head -1; } > "$OUT/toolchain.txt"
"$CUDA/bin/nvcc" "${FLAGS[@]}" -o "$SCR/pearl_c_sm120.cubin" "$ROOT/benchmarks/pouw/pearl_c_sm120/pearl_c_sm120.cu" \
  2> "$OUT/ptxas_pearl_c_sm120.txt"
"$CUDA/bin/nvcc" "${FLAGS[@]}" -o "$SCR/pd.cubin" "$HERE/pd.cu" 2> "$OUT/ptxas_pd.txt"
(cd "$SCR" && sha256sum ./*.cubin) > "$OUT/cubins.sha256"
"$CUDA/bin/cuobjdump" -sass "$SCR/pd.cubin" > "$SCR/pd.sass"
"$CUDA/bin/cuobjdump" -sass -fun pd_probe_3 "$SCR/pd.cubin" > "$OUT/pd_probe_3.sass"   # the rolled body, inlined
"$CUDA/bin/cuobjdump" -sass -fun pd_probe_19 "$SCR/pd.cubin" > "$OUT/pd_probe_19.sass"   # and the paired consumer's
python3 - "$SCR/pd.sass" > "$OUT/sass_ops.txt" <<'EOF'
import collections, re, sys
fn, ops = None, {}
for line in open(sys.argv[1]):
    if (m := re.match(r"\s+Function : (\S+)", line)):
        fn = m.group(1)
        ops[fn] = collections.Counter()
    elif fn and (m := re.match(r"\s+/\*[0-9a-f]+\*/\s+(?:@!?U?P\w+\s+)?([A-Z0-9_]+(?:\.WIDE)?)", line)):
        ops[fn][m.group(1)] += 1
for fn, c in ops.items():
    print(fn, sum(c.values()), " ".join(f"{k}:{v}" for k, v in c.most_common(14)))
EOF
cat "$OUT/sass_ops.txt"
grep -E "pd_|spill|stack" "$OUT/ptxas_pd.txt" | head -80 || true

g++ -O2 -std=c++20 -pthread -shared -fPIC -Wno-unknown-pragmas -o "$SCR/pd_host.so" "$HERE/pd_host.cpp"
PYTHONPATH="$ROOT" python3 "$HERE/host_test.py" "$SCR/pd_host.so" > "$OUT/host_test.txt" 2>&1
tail -3 "$OUT/host_test.txt"

export PD_SIDE=${PD_SIDE:-3,19,23}
B=(python3 "$HERE/bench.py" "$SCR/pearl_c_sm120.cubin" "$SCR/pd.cubin" --out "$OUT" --reps "$REPS" --rows "$ROWS" --shapes "$SHAPES")
gpu-lease 1 --wait --preemptible --max-min 25 -- bash -c '
  set -euo pipefail
  nvidia-smi --query-gpu=index,uuid,clocks.sm,clocks.max.sm,temperature.gpu,power.draw --format=csv > "$OUT/gpu.csv"
  "$@"
  "$0" profile -t cuda --cuda-graph-trace=node -o "$OUT/trace" -f true "$@" --trace > "$OUT/nsys.log" 2>&1
' "$CUDA/bin/nsys" "${B[@]}"

"$CUDA/bin/nsys" export --type sqlite --force-overwrite true -o "$OUT/trace.sqlite" "$OUT/trace.nsys-rep" > /dev/null
python3 "$HERE/trace.py" "$OUT" | tee "$OUT/trace.txt"
du -sh "$OUT"
