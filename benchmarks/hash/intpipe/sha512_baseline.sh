#!/usr/bin/env bash
# compute-accounting's SHA-512 committer (benchmarks/pouw/commit_gpu, built from its own path, unedited) on one leased GPU
# of whatever arch the machine has: raw compression rate, and hm96-sha512 4 KiB row-seg/v1 leaves in a frame-v3-sha512 tree
# end to end, each build checked bit for bit against reference.py in the same run before it is timed.
# cwd = the source tree's root. Outside the lease: the builds (BUILDS, default a0r0l, the README's fastest form, and a0r0,
# the straight-line one: its SASS is the per-block count), reference.py's cases (CHECK, default full) and parameter files,
# and SASS counts of sha_rate. Lease 1: selftest and check per build. Lease 2: the timing, builds interleaved rep by rep
# (ROUNDS outer rounds of `sha 20`, then 4 KiB leaves at 8192 leaves (o_proj's 32 MiB) and at LARGE_N leaves).
# Clocks are read (NVML in the bench lines, nvidia-smi every 200 ms), never set.
#
#   research run --on vy-dc-h200 --project verity --campaign hash --source . --cwd source --env GPU_LEASE_WHO=hash-intpipe \
#     -- bash benchmarks/hash/intpipe/sha512_baseline.sh
# LEASE_ARGS (default "--max-min 14"; on vy-nebius-2 "--preemptible --max-min 20"), ROUNDS (5), REPS (9), LARGE_N (65536).
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda/bin:$PATH
export D=benchmarks/pouw/commit_gpu I=benchmarks/hash/intpipe PY=python3 PYTHONPATH=.
W=$(mktemp -d /dev/shm/intpipe-sha512.XXXXXX)
export W
trap 'rm -rf "$W"' EXIT
git rev-parse HEAD > "$O/commit.txt" 2>&1
nproc > "$O/nproc.txt"
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
case $CC in 120) SM=120a ;; *) SM=$CC ;; esac
export SM
echo "compute_cap $CC sm_$SM" > "$O/arch.txt"

build() {   # gpu.sh's tags: a<SHA_ADD>r<SHA_ROT>[l] (l: SHA_ROLL=1)
  local t=$1 add=${1:1:1} rot flags="-DSHA_ROLL=0"
  rot=${t#a?r}
  case $rot in *l) rot=${rot%l}; flags="-DSHA_ROLL=1" ;; esac
  nvcc -O3 -std=c++17 -gencode arch=compute_$SM,code=sm_$SM -Xptxas -v -DSHA_ADD="$add" -DSHA_ROT="$rot" $flags \
    -o "$W/commit_gpu_$t" $D/commit_gpu.cu -lcublasLt -lnvidia-ml > "$O/build-$t.log" 2>&1
  echo "build $t rc $?" >> "$O/rc.txt"
}
export BUILDS=${BUILDS:-a0r0l a0r0}
for t in $BUILDS; do build "$t" & done
$PY $D/reference.py prepare "$W/cases" "${CHECK:-full}" > "$O/prepare.jsonl" 2> "$O/prepare.err"
echo "prepare rc $?" >> "$O/rc.txt"
export LARGE_N=${LARGE_N:-65536}
for n in 8192 "$LARGE_N"; do $PY $D/reference.py params "$W/p-seg-$n.bin" "$n" 32768 --seg; done
wait
for t in $BUILDS; do
  [ -x "$W/commit_gpu_$t" ] || continue
  cuobjdump -sass "$W/commit_gpu_$t" > "$W/sass-$t.txt" 2>&1
  case $t in *l) trip=4 ;; *) trip=1 ;; esac
  $PY $I/sass_count.py "$W/commit_gpu_$t" sha_rate --trip "$trip" > "$O/sass-$t.json" 2>> "$O/sass.err"
  gzip -c "$W/sass-$t.txt" > "$O/sass-$t.txt.gz"
done
cat "$O/rc.txt"

export REPS=${REPS:-9} ROUNDS=${ROUNDS:-5}
# shellcheck disable=SC2086,SC2016
gpu-lease 1 --wait ${LEASE_ARGS:---max-min 14} -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,compute_cap,clocks.sm,clocks.max.sm,power.limit --format=csv \
    -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  timed=""
  for t in $BUILDS; do
    [ -x "$W/commit_gpu_$t" ] || continue
    "$W/commit_gpu_$t" selftest >> "$O/selftest.jsonl"; s=$?
    $PY $D/reference.py check "$W/commit_gpu_$t" "$W/cases" "$t" > "$O/check-$t.jsonl" 2> "$O/check-$t.err"; c=$?
    echo "selftest $t rc $s; check $t rc $c" >> "$O/rc.txt"
    [ $s = 0 ] && [ $c = 0 ] && timed="$timed $t"
  done
  echo "$timed" > "$W/timed"
  echo "timed:$timed" >> "$O/rc.txt"'
echo "lease 1 rc $?" >> "$O/rc.txt"
export TIMED
TIMED=$(cat "$W/timed" 2>/dev/null)
# shellcheck disable=SC2086,SC2016
[ -n "$TIMED" ] && gpu-lease 1 --wait ${LEASE_ARGS:---max-min 14} -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,compute_cap,clocks.sm,clocks.max.sm,power.limit --format=csv \
    -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active \
    --format=csv -lms 200 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  for r in $(seq "$ROUNDS"); do
    for t in $TIMED; do "$W/commit_gpu_$t" sha 20 >> "$O/sha-$t.jsonl" 2>> "$O/bench.err"; done
    for t in $TIMED; do
      for n in 8192 $LARGE_N; do
        "$W/commit_gpu_$t" bench "$W/p-seg-$n.bin" seg "$n" 32768 "$REPS" >> "$O/bench-seg-$t.jsonl" 2>> "$O/bench.err"
        echo "round $r bench seg n=$n $t rc $?" >> "$O/rc.txt"
      done
    done
  done
  kill $smi'
echo "lease 2 rc $?" >> "$O/rc.txt"
cat "$O/rc.txt"
for f in "$O"/check-*.jsonl; do [ -e "$f" ] && tail -1 "$f"; done
$PY $I/summary.py sha512 "$O" | tee "$O/summary.json"
