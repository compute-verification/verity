#!/usr/bin/env bash
# sharaw on one leased GPU (cwd = the source tree's root). Outside the lease: circuits' native_rows.cu fetched at a pinned
# commit and built as its own library, sharaw built for the card, the reference's cases and the parameter files. Under one
# lease: the selftest, every sharaw form's bit-exactness check against the reference (check.py), then at each step size
# (DECODE_N, PREFILL_N leaves of 4 KiB) the bench: every committer's levels compared with commit_gpu's, then the raw rate,
# commit_gpu, native_rows and the sharaw forms interleaved rep by rep with the SM clock of each, then the time split.
# PROF=1: a second lease for nsys (each arm at both sizes) and ncu (the leaf kernels at the decode size). Clocks are read,
# never set.
#
#   research run --on vy-dc-h200 --project verity --campaign hash --source . --cwd source --env GPU_LEASE_WHO=hash-sha-raw \
#     -- bash benchmarks/hash/sharaw/gpu.sh
# REPS (default 7); LEASE_ARGS (e.g. --preemptible on vy-nebius-2); PROF (default 1)
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda/bin:$PATH PYTHONPATH=.
export D=benchmarks/hash/sharaw
W=$(mktemp -d)
export W
trap 'rm -rf "$W"' EXIT
git rev-parse HEAD > "$O/commit.txt" 2>&1
NR=de869b7d1f3f09dc5ae0fc098f99c719b1391cf1   # cursor/commit-speed-21f3 at 06:13Z, 18 Oct
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | head -1 | tr -d .)
GEN="-gencode arch=compute_${cc}a,code=sm_${cc}a"
echo "arch $cc" > "$O/rc.txt"
(
  mkdir -p "$W/nr" && git -C "$W/nr" init -q && git -C "$W/nr" fetch -q --depth=1 https://github.com/compute-verification/verity.git $NR &&
    for f in native_rows.cu sha512.cuh; do git -C "$W/nr" show $NR:integrations/vllm/verity_vllm/commit/committer/$f > "$W/nr/$f"; done &&
    sha256sum "$W/nr/native_rows.cu" "$W/nr/sha512.cuh" > "$O/native_rows.sha256" &&
    nvcc -O3 --use_fast_math -std=c++17 -shared -Xcompiler -fPIC $GEN -Xptxas -v -DVERITY_ROWS_MIN_BLOCKS=2 "$W/nr/native_rows.cu" \
      -o "$W/native_rows.so" > "$O/build-native.log" 2>&1
  echo "build native rc $?" >> "$O/rc.txt"
) &
nvcc -O3 -std=c++17 $GEN -Xptxas -v -o "$W/sharaw" $D/sharaw.cu -lnvidia-ml -ldl > "$O/build.log" 2>&1
echo "build sharaw rc $?" >> "$O/rc.txt"
python3 $D/check.py prepare "$W/cases" > "$O/prepare.jsonl" 2> "$O/prepare.err"
echo "prepare rc $?" >> "$O/rc.txt"
export DECODE_N=${DECODE_N:-34737} PREFILL_N=${PREFILL_N:-3293653}   # 142.28 MB and 13.49 GB a step (r20261018-031955-d2b3)
for n in $DECODE_N $PREFILL_N; do python3 benchmarks/pouw/commit_gpu/reference.py params "$W/p-$n.bin" "$n" 32768 --ks-seed 11; done
wait
cat "$O/rc.txt"
[ -x "$W/sharaw" ] || exit 1
export SO="$W/native_rows.so"
[ -f "$SO" ] || SO=""
export REPS=${REPS:-7}
# shellcheck disable=SC2016,SC2086
gpu-lease 1 --wait --max-min 14 ${LEASE_ARGS:-} -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,power.limit,memory.used --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  "$W/sharaw" selftest > "$O/selftest.jsonl"; s=$?
  ok=$s
  for f in 0 1 2 3 4 5 6 7 8 9 10 11 12; do
    SHARAW_FORM=$f python3 $D/check.py check "$W/sharaw" "$W/cases" form$f > "$O/check-form$f.jsonl" 2> "$O/check-form$f.err"; c=$?
    echo "check form$f rc $c" >> "$O/rc.txt"
    [ $c = 0 ] || ok=1
  done
  echo "selftest rc $s; checks ok=$ok" >> "$O/rc.txt"
  [ $s = 0 ] || exit 1   # a form that fails the reference is not fatal: bench compares every arm with commit_gpu and drops a differing one
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu --format=csv -lms 200 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  for n in $DECODE_N $PREFILL_N; do
    "$W/sharaw" bench "$W/p-$n.bin" $n $REPS $SO > "$O/bench-$n.jsonl" 2> "$O/bench-$n.err"
    echo "bench $n rc $?" >> "$O/rc.txt"
  done
  kill $smi'
echo "lease rc $?" >> "$O/rc.txt"
if [ "${PROF:-1}" = 1 ]; then
  # shellcheck disable=SC2016,SC2086
  gpu-lease 1 --wait --max-min 12 ${LEASE_ARGS:-} -- bash -c '
    for n in $DECODE_N $PREFILL_N; do
      for a in commit_gpu native_rows v3_salt_s2 v3s_fold; do
        [ $a = native_rows ] && [ -z "$SO" ] && continue
        nsys profile -t cuda -o "$W/t-$a-$n" --force-overwrite true "$W/sharaw" nsys "$W/p-$n.bin" $n $a $SO > /dev/null 2>> "$O/nsys.err"
        nsys stats -r cuda_gpu_trace -f csv -o "$O/nsys-$a-$n" "$W/t-$a-$n.nsys-rep" >> "$O/nsys.err" 2>&1
        echo "nsys $a $n rc $?" >> "$O/rc.txt"
      done
    done
    for a in commit_gpu native_rows v3_salt_s2 v3s_fold; do
      [ $a = native_rows ] && [ -z "$SO" ] && continue
      ncu --set basic --section WarpStateStats --section SchedulerStats --section Occupancy --section ComputeWorkloadAnalysis \
        --section MemoryWorkloadAnalysis -k regex:"salt|leaf|level|node|sr_" --launch-count 6 --csv --page details \
        "$W/sharaw" nsys "$W/p-$DECODE_N.bin" $DECODE_N $a $SO > "$O/ncu-$a.csv" 2> "$O/ncu-$a.err"
      echo "ncu $a rc $?" >> "$O/rc.txt"
    done'
  echo "prof lease rc $?" >> "$O/rc.txt"
fi
cat "$O/rc.txt"
for f in "$O"/check-*.jsonl; do [ -e "$f" ] && tail -1 "$f"; done
grep -h '"arm"' "$O"/bench-*.jsonl | cut -c1-200
