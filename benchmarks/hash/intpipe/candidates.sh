#!/usr/bin/env bash
# The integer-pipe candidates (candidates.cu) against full SHA-512 on one leased GPU of whatever arch the machine has.
# cwd = the source tree's root. Outside the lease: one binary a candidate and form (CANDS, FORMS: 1 rolled, 0 straight-line;
# built in parallel), SASS counts of each raw-rate kernel, the check operand (CHECK_N 4 KiB leaves). Lease 1: each binary's
# dump. Then check.py on every dump (CPU). Lease 2: every checked binary's bench (REPS reps of the raw rate and of the
# 4 KiB-leaf tree over each N leaves, baseline and candidate back to back), with nvidia-smi's clocks every 200 ms.
#
#   research run --on vy-dc-h200 --project verity --campaign hash --source . --cwd source --env GPU_LEASE_WHO=hash-intpipe \
#     -- bash benchmarks/hash/intpipe/candidates.sh
# LEASE_ARGS (default "--max-min 14"; on vy-nebius-2 "--preemptible --max-min 20"), REPS (15), N (65536,262144: 256 MiB
# and 1 GiB), CHECK_N (1024), FETCH (128: the bytes of each leaf stream the loader reads at a time); FETCHES="128 256"
# runs the whole script once a value, one after another, into $OUT/f<FETCH>.
set -uo pipefail
if [ -n "${FETCHES:-}" ]; then
  for F in $FETCHES; do
    mkdir -p "$OUT/f$F"
    OUT=$OUT/f$F FETCH=$F FETCHES='' bash "$0"
  done
  exit 0
fi
export O=${OUT:?}
export PATH=/usr/local/cuda/bin:$PATH
export I=benchmarks/hash/intpipe PY=python3 PYTHONPATH=.
W=$(mktemp -d /dev/shm/intpipe-cand.XXXXXX)
export W
trap 'rm -rf "$W"' EXIT
git rev-parse HEAD > "$O/commit.txt" 2>&1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
case $CC in 120) SM=120a ;; *) SM=$CC ;; esac
echo "compute_cap $CC sm_$SM" > "$O/arch.txt"
# name:kind:rounds:rolled-loop trip (sass_count's inner-loop count in the rolled form)
CANDS=${CANDS:-sha512:0:80:4 sha512r80:1:80:4 sha512r64:1:64:3 sha512r48:1:48:2 sha512r40:1:40:1 sha512r32:1:32:1 sha512r24:1:24:1 sha256:2:64:3 blake3:3:7:1 blake3chunks:4:7:1 xor:5:0:1}
FORMS=${FORMS:-1 0}
for c in $CANDS; do
  IFS=: read -r name kind r _ <<< "$c"
  for f in $FORMS; do
    ( nvcc -O3 -std=c++17 -gencode arch=compute_$SM,code=sm_$SM -Xptxas -v -DCAND_KIND="$kind" -DCAND_R="$r" -DSHA_ROLL="$f" -DFETCH="${FETCH:-128}" \
        -o "$W/cand-$name-r$f" $I/candidates.cu -lnvidia-ml > "$O/build-$name-r$f.log" 2>&1
      echo "build $name r$f rc $?" >> "$O/rc.txt" ) &
  done
done
$PY $I/check.py operand "$W/operand.bin" "${CHECK_N:-1024}"
wait
BINS=""
for c in $CANDS; do
  IFS=: read -r name kind r trip <<< "$c"
  for f in $FORMS; do
    b="$W/cand-$name-r$f"
    [ -x "$b" ] || continue
    BINS="$BINS $name-r$f"
    t=1; [ "$f" = 1 ] && t=$trip
    $PY $I/sass_count.py "$b" rate_cand --trip "$t" > "$O/sass-$name-r$f.json" 2>> "$O/sass.err"
    $PY $I/sass_count.py "$b" rate_base --trip "$([ "$f" = 1 ] && echo 4 || echo 1)" > "$O/sass-base-$name-r$f.json" 2>> "$O/sass.err"
    cuobjdump -sass "$b" 2>&1 | gzip > "$O/sass-$name-r$f.txt.gz"
  done
done
export BINS CHECK_N=${CHECK_N:-1024}
cat "$O/rc.txt"
# shellcheck disable=SC2086,SC2016
gpu-lease 1 --wait ${LEASE_ARGS:---max-min 14} -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,compute_cap,clocks.sm,clocks.max.sm,power.limit --format=csv \
    -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  for b in $BINS; do
    mkdir -p "$W/dump-$b"
    "$W/cand-$b" dump "$W/dump-$b" "$W/operand.bin" "$CHECK_N" 2>> "$O/dump.err"
    echo "dump $b rc $?" >> "$O/rc.txt"
  done'
echo "lease 1 rc $?" >> "$O/rc.txt"
for b in $BINS; do
  ( $PY $I/check.py check "$W/dump-$b" "$W/operand.bin" > "$O/check-$b.json" 2> "$O/check-$b.err"
    echo "check $b rc $?" >> "$O/rc.txt" ) &
done
wait
CHECKED=""
for b in $BINS; do grep -q '"ok": true}$' "$O/check-$b.json" 2>/dev/null && CHECKED="$CHECKED $b"; done
echo "checked:$CHECKED" >> "$O/rc.txt"
export CHECKED REPS=${REPS:-15} N=${N:-65536,262144}
# shellcheck disable=SC2086,SC2016
[ -n "$CHECKED" ] && gpu-lease 1 --wait ${LEASE_ARGS:---max-min 14} -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,compute_cap,clocks.sm,clocks.max.sm,power.limit --format=csv \
    -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active \
    --format=csv -lms 200 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  for b in $CHECKED; do
    "$W/cand-$b" bench "$REPS" "$N" >> "$O/bench.jsonl" 2>> "$O/bench.err"
    echo "bench $b rc $?" >> "$O/rc.txt"
  done
  kill $smi'
echo "lease 2 rc $?" >> "$O/rc.txt"
cat "$O/rc.txt"
$PY $I/summary.py hashes "$O" | tee "$O/summary.jsonl"
