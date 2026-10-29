#!/usr/bin/env bash
# ssn_1l's kernel on one H200 in one gpu-lease: built and SASS-gated (IGMMA in every ssn_k, no local memory) before the
# lease; under it the exact check of every word, then the timings at each M, the SM clock sampled beside it all.
# Measurement only. cwd: the source tree's root.
#   MS (64 512 8192: Llama-3.2-1B's T 512 at 64 and 512 rows, and T 8,192; an M under 64 is one decode call over its
#   padded tile, e.g. "1 16"), REPS (5), BUDGET_GB (16: the exported words a
#   hash chunk holds), ROWS (0 65536 16384: the hash pass's row bytes, 0 for row-seg, -B for L1's plain leaves of B-byte
#   rows), CUDA_HOME (/usr/local/cuda-13.0),
#   BUILD_ONLY (0; 1 stops after the build and the SASS gate, on any machine with a CUDA toolkit: no lease, no GPU),
#   LEASE_MIN (25), CHECK (1; 0 skips the exact check), TIME (1; 0 skips the timings), NOT_AFTER (a UTC time past which
#   the leased part does no GPU work), CAPS (0: beside's caps on a row hash launch's blocks, each a pass
#   over MS; time_m<M>_cap<C>.jsonl), ROOT_A (0; 1 puts root(A_j), its leaves and the tree above them, in the forward ahead
#   of each call's feed, by the words' scheme or by ROOT_A_ROWS's: row bytes, 0 row-seg over A_j's rows), ROOT_A_LIST (the
#   schemes root(A_j) is timed by a call; default ROWS), WORDS_TREE (0; 1 builds the tree above each chunk's leaves in the
#   forwards), PRIO (0; 1 runs the forwards' GEMM stream at the greatest stream priority, the hash streams at the least),
#   SMALL_THREADS (unset; N builds native_rows.cu a second time with blocks of N threads, a build-time edit of a copy, for
#   the leaves of any launch of fewer than SMALL_ROWS_MAX rows), TAG (0; 1 times the ssn-1l-tag layout; the check runs both),
#   L1 (0; 1 puts L1 + (a)'s root(A_j) before each call's feed and forming; 2 the same root shaped for latency),
#   L1_CTA (64; L1=2's least CTA size, 64 or 128),
#   BLC (0; 1 times blc4a-t64-e7: the tag layout with three lane coins a 64-row tile in each later step, streamed after A_j;
#   2 blc4a-t64-e7-s1, the same with lane coins on step 1 too; the check runs all four layouts), BLC_W (1:1:1: A' in the
#   three lane-coin lanes, e.g. 1:127:1; the check runs under it too), ROT (0; 1 with BLC times blc4a-t64-e7-rot, A' in
#   the lane-coin lanes (127, 1, 1) on even rows and (1, 127, 1) on odd ones in place of BLC_W; the check runs it on
#   blc4a-t64-e7-s1 without XU, with XU and with FOLD), XU (0; 1 times ssn_1l_tag_xu on the tag layout:
#   its last step and the GEMMs' xu epilogue serving bf16; the check runs it on the three tag layouts against stock's
#   epilogue), FOLD (0; 1 with BLC, not XU, times the lead served noisy: the GEMMs' xu epilogue folding out the lane
#   coins alone), EPI (0; 1 the GEMMs run stock's epilogue on the final words),
#   SPLIT (0; R runs beside's GEMM stream in a green context of R SMs and its hash streams in one of the rest; no serial),
#   BARE (0; 1 also times beside with nothing ahead of each call's GEMM, the schedule's floor, and beside's GEMM stream
#   alone, no hash), NBUF (8: the most buffers beside's hash streams take, each its own stream), VARIANTS (unset; else
#   passes of the timings, each a comma-separated NAME=VALUE list over the environment above, e.g. "L1=1 L1=2,PRIO=1":
#   time_m<M>_<the list with = as - and , as _>.jsonl).
#   VERITY_ROWS_PHASES and VERITY_ROWS_FORM reach the served committer (native_rows.cu).
set -uo pipefail
export OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}; mkdir -p "$OUT"
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$OUT/steps.txt"; }
HERE=benchmarks/pouw/np_kernel

if [ "${1:-}" != leased ]; then
  BUILD_DIR=$(mktemp -d /tmp/np-kernel-XXXXXX); export BUILD_DIR
  trap 'rm -rf "$BUILD_DIR"' EXIT
  CUDA=${CUDA_HOME:-/usr/local/cuda-13.0}
  say "build $(git rev-parse --short=10 HEAD 2>/dev/null) into $BUILD_DIR"
  COMMITTER=integrations/vllm/verity_vllm/commit/committer
  python3 "$HERE/plain_rows.py" "$COMMITTER/native_rows.cu" "$BUILD_DIR/native_rows_p.cu" || { say "plain_rows.py failed"; exit 3; }
  SRCS=("$HERE/ssn.cu" "$COMMITTER/native_rows.cu" "$BUILD_DIR/native_rows_p.cu" -I "$COMMITTER")
  if [ -n "${SMALL_THREADS:-}" ]; then
    # the copy keeps the leaf kernels' register cap (128: threads x min blocks = 512) and renames its symbols and sha512.cuh's
    sed -e "s/^constexpr int THREADS = 256;/constexpr int THREADS = $SMALL_THREADS;/" \
      -e 's/^#define VERITY_ROWS_MIN_BLOCKS 2$/#define VERITY_ROWS_MIN_BLOCKS (512 \/ THREADS)/' \
      -e 's/__launch_bounds__(THREADS, 2)/__launch_bounds__(THREADS, 512 \/ THREADS)/' \
      -e 's/\bverity_rows_/verity_rows_s_/g' -e 's/^#include "sha512.cuh"/#define sha sha_s\n#include "sha512.cuh"/' \
      "$COMMITTER/native_rows.cu" > "$BUILD_DIR/native_rows_s.cu"
    n=$(grep -c -e "THREADS = $SMALL_THREADS;" -e 'MIN_BLOCKS (512 / THREADS)' -e 'THREADS, 512 / THREADS' \
      -e '^#define sha sha_s' "$BUILD_DIR/native_rows_s.cu")
    [ "$n" = 4 ] || { say "the small-block copy of native_rows.cu took $n of its 4 edits"; exit 3; }
    SRCS+=("$BUILD_DIR/native_rows_s.cu" -DSSN_SMALL_ROWS="$SMALL_THREADS" -I "$COMMITTER")
  fi
  "$CUDA/bin/nvcc" -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -Xptxas -v "${SRCS[@]}" -lcuda \
    -o "$BUILD_DIR/ssn" > "$OUT/build_ssn.log" 2>&1
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
  [ "${BUILD_ONLY:-0}" = 1 ] && { say "BUILD_ONLY: no lease, no GPU work"; exit 0; }
  gpu-lease status 2>&1 | tee -a "$OUT/steps.txt"
  say "lease requested"
  gpu-lease 1 --wait --mem-gb 0 --max-min "${LEASE_MIN:-25}" -- bash "$0" leased
  rc=$?; say "lease rc $rc"
  exit "$rc"
fi

say "leased CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"
if [ -n "${NOT_AFTER:-}" ] && [ "$(date -u +%s)" -gt "$(date -u -d "$NOT_AFTER" +%s)" ]; then
  say "past NOT_AFTER $NOT_AFTER: no GPU work"; exit 5
fi
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
[ "${TIME:-1}" = 1 ] || exit "$worst"
for v in ${VARIANTS:--}; do
  vars=(); [ "$v" = - ] || IFS=, read -ra vars <<< "$v"
  for cap in ${CAPS:-0}; do
    for m in ${MS:-64 512 8192}; do
      f=time_m$m; [ "$cap" = 0 ] || f=${f}_cap$cap; [ "$v" = - ] || f=${f}_$(tr = - <<< "${v//,/_}")
      # shellcheck disable=SC2086
      env "${vars[@]}" HASH_CAP=$cap "$BUILD_DIR/ssn" time "${REPS:-5}" "$m" "${BUDGET_GB:-16}" ${ROWS:-0 65536 16384} \
        > "$OUT/$f.jsonl" 2> "$OUT/$f.err"
      rc=$?; say "time M $m cap $cap ${vars[*]} rc $rc ($(wc -l < "$OUT/$f.jsonl") lines)"
      [ "$rc" = 0 ] || { worst=$rc; tail -5 "$OUT/$f.err"; }
    done
  done
done
exit "$worst"
