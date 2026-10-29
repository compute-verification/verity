#!/usr/bin/env bash
# The performance gate on one leased GPU, run by gpu.sh (BENCH=1) after the conformance, with its O, W and VC_LIB. Outside the
# lease: sharaw.cu with the commit_gpu it includes at their pinned commit and native_rows.cu at its own, fetched from GitHub;
# native_rows.so built with sharaw's flags; rows.cu and pearl.cu built for this card against libvc. Under one lease: rows.cu at
# each step (DECODE_N and GIB_N rows of 4 KiB: 142 MB and 1 GiB; MID_N between, where AUTO switches form), then pearl.cu
# (Pearl-C's decode shapes). Each prints JSON lines: every arm's output compared with the replaced code's first, then the arms
# interleaved rep by rep with the SM clock of each.
set -uo pipefail
: "${O:?}" "${W:?}" "${VC_LIB:?}"
D=benchmarks/hash/lib
SR=4369ae628de156b7d9115a0e1ad6cf107c132a9e   # sharaw (benchmarks/hash/sharaw), commit_gpu unchanged since
NR=934d0bcbdf3f838d51696b78d34b85235e44e931   # native_rows.cu as libvc vendored it
URL=https://github.com/compute-verification/verity.git
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | head -1 | tr -d .)
GEN="-gencode arch=compute_${cc}a,code=sm_${cc}a"
LIBDIR=$(dirname "$VC_LIB")
B="$W/bench"
mkdir -p "$B/sr" "$B/nr"
{
  git -C "$B/sr" init -q && git -C "$B/sr" fetch -q --depth=1 $URL $SR &&
    git -C "$B/sr" checkout -q FETCH_HEAD -- benchmarks/hash/sharaw benchmarks/pouw/commit_gpu
  echo "fetch sharaw rc $?"
  git -C "$B/nr" init -q && git -C "$B/nr" fetch -q --depth=1 $URL $NR &&
    for f in native_rows.cu sha512.cuh; do git -C "$B/nr" show $NR:integrations/vllm/verity_vllm/commit/committer/$f > "$B/nr/$f"; done
  echo "fetch native_rows rc $?"
} >> "$O/rc.txt" 2>&1
sha256sum "$B/sr/benchmarks/hash/sharaw/sharaw.cu" "$B/sr/benchmarks/pouw/commit_gpu/commit_gpu.cuh" "$B/nr/native_rows.cu" > "$O/bench-sources.sha256" 2>&1
# sharaw's flags, on the shared runtime as libvc is; rows.cu's Nr calls its launchers (sharaw's own Native predates max_blocks)
nvcc -O3 --use_fast_math -std=c++17 -shared -Xcompiler -fPIC $GEN -cudart shared -Xptxas -v -DVERITY_ROWS_MIN_BLOCKS=2 \
  "$B/nr/native_rows.cu" -o "$B/native_rows.so" > "$O/build-native.log" 2>&1 &
nvcc -O3 -std=c++17 $GEN -cudart shared -Xptxas -v -DSHARAW_CU="\"$B/sr/benchmarks/hash/sharaw/sharaw.cu\"" $D/bench/rows.cu \
  -o "$B/vc_rows" -L"$LIBDIR" -lvc -Xlinker -rpath="$LIBDIR" -lnvidia-ml -ldl > "$O/build-bench-rows.log" 2>&1 &
nvcc -O3 -std=c++17 $GEN -cudart shared -Xptxas -v $D/bench/pearl.cu -o "$B/vc_pearl" -L"$LIBDIR" -lvc -Xlinker -rpath="$LIBDIR" \
  -lnvidia-ml > "$O/build-bench-pearl.log" 2>&1 &
export DECODE_N=${DECODE_N:-34737} GIB_N=${GIB_N:-262144} MID_N=${MID_N:-65536 131072} REPS=${REPS:-9}
for n in $DECODE_N $MID_N $GIB_N; do uv run python benchmarks/pouw/commit_gpu/reference.py params "$B/p-$n.bin" "$n" 32768 --ks-seed 11; done \
  > "$O/bench-params.log" 2>&1
wait
for f in native_rows.so vc_rows vc_pearl; do [ -e "$B/$f" ]; echo "build $f rc $?"; done >> "$O/rc.txt"
[ -x "$B/vc_rows" ] && [ -x "$B/vc_pearl" ] || { tail -30 "$O"/build-bench-*.log; exit 1; }
export B
# shellcheck disable=SC2016,SC2086
gpu-lease 1 --wait --max-min 14 ${LEASE_ARGS:-} -- bash -c '
  q="index,uuid,name,clocks.sm,clocks.max.sm,clocks.mem,power.draw,power.limit,temperature.gpu,clocks_throttle_reasons.active"
  nvidia-smi --query-gpu=$q --format=csv -i "$GPU_LEASE_UUID" > "$O/bench-gpu-before.txt" 2>&1
  for n in $DECODE_N $MID_N $GIB_N; do
    "$B/vc_rows" rows "$B/p-$n.bin" "$n" "$REPS" "$B/native_rows.so" > "$O/bench-rows-$n.jsonl" 2> "$O/bench-rows-$n.err"
    echo "bench rows $n rc $?" >> "$O/rc.txt"
  done
  "$B/vc_pearl" pearl "$REPS" > "$O/bench-pearl.jsonl" 2> "$O/bench-pearl.err"
  echo "bench pearl rc $?" >> "$O/rc.txt"
  nvidia-smi --query-gpu=$q --format=csv -i "$GPU_LEASE_UUID" > "$O/bench-gpu-after.txt" 2>&1'
echo "bench lease rc $?" >> "$O/rc.txt"
grep -h '"same_levels"\|"same_bytes"' "$O"/bench-*.jsonl | grep -v '"differing": 0\|"ok": true' | head -20
grep -h '"arm"' "$O"/bench-*.jsonl | python3 -c '
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    print(r.get("piece", "rows"), r.get("shape", r.get("n")), r["arm"], r.get("ms", r.get("us")), r["mhz"])'
