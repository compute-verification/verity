#!/usr/bin/env bash
# exact_sweep.py's epilogue folds on an H200 (epilogue_gpu.cu): the SASS of the stock, magic and column-fold epilogue
# kernels, each kernel's opcode counts and their arithmetic ops (sass.json), then on the device the bit-equality sweep
# (device.json). From a pushed commit:
#
#   research run --on vy-dc-h200-5 --project verity --cwd source --timeout 30m -- bash benchmarks/pouw/theory/epilogue_gpu.sh
#
# The build (sm_90a) and the SASS dump run outside the lease; the sweep runs under one gpu-lease (or the GPU the queue
# gave it, QUEUED=1). Exit: 0 done, 1 build failed, 2 no nvcc or a CUDA error, 3 a fold not bit-exact or an edge not
# caught.
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [ "${EPI_LEASED:-}" != 1 ]; then
  mkdir -p "$OUT/sass" "$OUT/env" "$OUT/bin"
  NVCC=${NVCC:-}
  for c in /usr/local/cuda-13.0 /usr/local/cuda; do [ -z "$NVCC" ] && [ -x "$c/bin/nvcc" ] && NVCC=$c/bin/nvcc; done
  [ -n "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "epilogue_gpu.sh: no nvcc" >&2; exit 2; }
  "$NVCC" --version >"$OUT/env/nvcc.txt" 2>&1
  BIN="$OUT/bin/epilogue_gpu"
  "$NVCC" -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -Xptxas -v "$HERE/epilogue_gpu.cu" -o "$BIN" \
    >"$OUT/build.log" 2>&1 || { cat "$OUT/build.log" >&2; exit 1; }
  sha256sum "$BIN" >"$OUT/bin/epilogue_gpu.sha256"
  "$(dirname "$NVCC")/cuobjdump" -sass "$BIN" >"$OUT/sass/epilogue_gpu.sass" 2>&1
  python3 - "$OUT/sass/epilogue_gpu.sass" >"$OUT/sass.json" <<'PY'
import collections, json, re, sys

ARITH = re.compile(r"^(I2F|F2F|FMUL|FFMA|FADD|IADD3|IMAD|LOP3|SHF)")
kernels, cur = {}, None
for line in open(sys.argv[1]):
    m = re.search(r"Function : (\S+)", line)
    if m:
        cur = m.group(1) if m.group(1).startswith("epi_") else None
        if cur:
            kernels[cur] = collections.Counter()
        continue
    m = re.search(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)", line)
    if cur and m:
        kernels[cur][m.group(1)] += 1
out = {k: {"opcodes": dict(sorted(c.items())), "arith": dict(sorted((o, n) for o, n in c.items() if ARITH.match(o)))}
       for k, c in sorted(kernels.items())}
json.dump(out, sys.stdout, indent=1)
print()
PY
  cat "$OUT/sass.json"
  lease=()
  command -v gpu-lease >/dev/null && lease=(gpu-lease 1 --wait --max-min 10 --)
  [ "${QUEUED:-0}" = 1 ] && lease=()
  [ ${#lease[@]} -eq 0 ] && echo "epilogue_gpu.sh: no gpu-lease (or queued); running on the GPU it was given" >&2
  "${lease[@]}" env EPI_LEASED=1 BIN="$BIN" bash "${BASH_SOURCE[0]}"
  rc=$?
  [ "$rc" = 1 ] && rc=3
  exit $rc
fi

# ---- leased ------------------------------------------------------------------------------------------------------------
nvidia-smi --query-gpu=name,uuid,compute_cap,clocks.sm,clocks.max.sm --format=csv >"$OUT/env/gpu.csv" 2>&1
"$BIN" "${EPI_BATCHES:-16}" | tee "$OUT/device.json"
exit "${PIPESTATUS[0]}"
