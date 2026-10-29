#!/usr/bin/env bash
# fold_rate.cu on one H200 in one gpu-lease: built, its SASS opcodes counted a kernel before the lease, then each
# variant timed. Measurement only. cwd: the source tree's root.
#   STEPS (4096), REPS (7), CUDA_HOME (/usr/local/cuda-13.0), BUILD_ONLY (0; 1 stops after the build and the count),
#   LEASE_MIN (5), NOT_AFTER (a UTC time past which the leased part does no GPU work)
set -uo pipefail
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$OUT/steps.txt"; }
HERE=benchmarks/pouw/hash_sparse

if [ "${1:-}" != leased ]; then
  BUILD_DIR=$(mktemp -d /tmp/fold-rate-XXXXXX); export BUILD_DIR
  trap 'rm -rf "$BUILD_DIR"' EXIT
  CUDA=${CUDA_HOME:-/usr/local/cuda-13.0}
  "$CUDA/bin/nvcc" -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -Xptxas -v "$HERE/fold_rate.cu" \
    -o "$BUILD_DIR/fold_rate" > "$OUT/build.log" 2>&1
  rc=$?; say "build rc $rc"; [ "$rc" = 0 ] || { tail -40 "$OUT/build.log"; exit 3; }
  "$CUDA/bin/cuobjdump" -sass "$BUILD_DIR/fold_rate" > "$OUT/fold_rate.sass"
  python3 - "$OUT/fold_rate.sass" > "$OUT/sass_counts.json" <<'EOF'
import collections, json, re, sys
funcs, cur = {}, None
for line in open(sys.argv[1]):
    m = re.search(r"Function : (\S+)", line)
    if m:
        cur = m.group(1); funcs[cur] = collections.Counter()
    elif cur:
        m = re.search(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_]*(?:\.[A-Z0-9_]+)*)", line)
        if m:
            funcs[cur][m.group(1).split(".")[0]] += 1
        funcs[cur]["local"] += bool(re.search(r"\b(LDL|STL)\b", line))
keep = ("IMMA", "LOP3", "IMAD", "IADD3", "LDG", "local")
print(json.dumps({k: {o: v[o] for o in keep} for k, v in funcs.items()}))
bad = [k for k, v in funcs.items() if v["local"]]
sys.exit(1 if bad else 0)
EOF
  rc=$?; say "SASS count rc $rc"; cat "$OUT/sass_counts.json"; [ "$rc" = 0 ] || exit 4
  [ "${BUILD_ONLY:-0}" = 1 ] && { say "BUILD_ONLY: no lease, no GPU work"; exit 0; }
  gpu-lease status 2>&1 | tee -a "$OUT/steps.txt"
  say "lease requested"
  gpu-lease 1 --wait --mem-gb 0 --max-min "${LEASE_MIN:-5}" -- bash "$0" leased
  rc=$?; say "lease rc $rc"
  exit "$rc"
fi

say "leased CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"
if [ -n "${NOT_AFTER:-}" ] && [ "$(date -u +%s)" -gt "$(date -u -d "$NOT_AFTER" +%s)" ]; then
  say "past NOT_AFTER $NOT_AFTER: no GPU work"; exit 5
fi
nvidia-smi --query-gpu=name,uuid,clocks.sm,clocks.max.sm,power.limit,memory.used --format=csv | tee -a "$OUT/steps.txt"
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,utilization.gpu --format=csv -lms 200 \
  -i "${CUDA_VISIBLE_DEVICES:-0}" > "$OUT/smi.csv" 2>&1 &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT
"$BUILD_DIR/fold_rate" "${STEPS:-4096}" "${REPS:-7}" > "$OUT/fold_rate.jsonl" 2> "$OUT/fold_rate.err"
rc=$?; say "fold_rate rc $rc"; cat "$OUT/fold_rate.jsonl"
exit "$rc"
