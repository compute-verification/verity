#!/usr/bin/env bash
# ssn_1l's kernel on one H200 in one gpu-lease: built and SASS-gated (IGMMA in every ssn_k, no local memory) before the
# lease; under it the exact check of every word, then the timings at each M, the SM clock sampled beside it all.
# Measurement only. cwd: the source tree's root.
#   MS (64 512 8192: Llama-3.2-1B's T 512 at 64 and 512 rows, and T 8,192), REPS (5), BUDGET_GB (16: the exported words a
#   hash chunk holds), ROWS (0 65536 16384: the hash pass's row bytes, 0 for row-seg), CUDA_HOME (/usr/local/cuda-13.0),
#   LEASE_MIN (25), CHECK (1; 0 skips the exact check), CAPS (0: beside's caps on a row hash launch's blocks, each a pass
#   over MS; time_m<M>_cap<C>.jsonl)
set -uo pipefail
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$OUT/steps.txt"; }
HERE=benchmarks/pouw/np_kernel

if [ "${1:-}" != leased ]; then
  BUILD_DIR=$(mktemp -d /tmp/np-kernel-XXXXXX); export BUILD_DIR
  trap 'rm -rf "$BUILD_DIR"' EXIT
  CUDA=${CUDA_HOME:-/usr/local/cuda-13.0}
  say "build $(git rev-parse --short=10 HEAD 2>/dev/null) into $BUILD_DIR"
  "$CUDA/bin/nvcc" -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -Xptxas -v "$HERE/ssn.cu" \
    integrations/vllm/verity_vllm/commit/committer/native_rows.cu -o "$BUILD_DIR/ssn" > "$OUT/build_ssn.log" 2>&1
  rc=$?; say "build ssn rc $rc"; [ "$rc" = 0 ] || { tail -40 "$OUT/build_ssn.log"; exit 3; }
  "$CUDA/bin/cuobjdump" -sass "$BUILD_DIR/ssn" > "$OUT/ssn.sass"
  python3 - "$OUT/ssn.sass" > "$OUT/gate_ssn.json" <<'EOF'
import json, re, sys
funcs, cur = {}, None
for line in open(sys.argv[1]):
    m = re.search(r"Function : (\S+)", line)
    if m:
        cur = m.group(1); funcs[cur] = {"IGMMA": 0, "local": 0}
    elif cur:
        funcs[cur]["IGMMA"] += "IGMMA" in line
        funcs[cur]["local"] += bool(re.search(r"\b(LDL|STL)\b", line))
gemms = {k: v for k, v in funcs.items() if "ssn_k" in k}
ok = len(gemms) == 4 and all(v["IGMMA"] > 0 and v["local"] == 0 for v in gemms.values())
print(json.dumps({"ok": ok, "functions": funcs}))
sys.exit(0 if ok else 1)
EOF
  rc=$?; say "SASS gate rc $rc"; [ "$rc" = 0 ] || { cat "$OUT/gate_ssn.json"; exit 4; }
  gpu-lease status 2>&1 | tee -a "$OUT/steps.txt"
  say "lease requested"
  gpu-lease 1 --wait --mem-gb 0 --max-min "${LEASE_MIN:-25}" -- bash "$0" leased
  rc=$?; say "lease rc $rc"
  exit "$rc"
fi

say "leased CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"
nvidia-smi --query-gpu=name,uuid,clocks.sm,clocks.max.sm,power.limit,memory.used --format=csv | tee -a "$OUT/steps.txt"
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,utilization.gpu,temperature.gpu,memory.used --format=csv -lms 200 \
  -i "${CUDA_VISIBLE_DEVICES:-0}" > "$OUT/smi.csv" 2>&1 &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT
worst=0
if [ "${CHECK:-1}" = 1 ]; then
  "$BUILD_DIR/ssn" check > "$OUT/check.jsonl" 2> "$OUT/check.err"
  worst=$?; say "check rc $worst"; cat "$OUT/check.jsonl"
fi
for cap in ${CAPS:-0}; do
  for m in ${MS:-64 512 8192}; do
    f=time_m$m; [ "$cap" = 0 ] || f=time_m${m}_cap$cap
    # shellcheck disable=SC2086
    HASH_CAP=$cap "$BUILD_DIR/ssn" time "${REPS:-5}" "$m" "${BUDGET_GB:-16}" ${ROWS:-0 65536 16384} > "$OUT/$f.jsonl" 2> "$OUT/$f.err"
    rc=$?; say "time M $m cap $cap rc $rc ($(wc -l < "$OUT/$f.jsonl") lines)"
    [ "$rc" = 0 ] || { worst=$rc; tail -5 "$OUT/$f.err"; }
  done
done
exit "$worst"
