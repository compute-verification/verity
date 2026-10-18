#!/usr/bin/env bash
# Latency levers for full SHA-512 in hm96 format at decode-sized windows (hm96lat.cu), on one leased GPU of whatever arch
# the machine has. cwd = the source tree's root.
#   1. hm96.sh, base SHA-512 only, into $OUT/leaf4k (4 KiB leaves at every window and batched size), $OUT/leaf2k (2 KiB
#      leaves of two 1 KiB segments: Verity's row-seg/v1, held to hm96_ref) and $OUT/leaf2k-seg512 (2 KiB leaves of four
#      512-byte segments: FORMAT CHANGE).
#   2. hm96lat: lease 1, `verify` at CHECK_N and every size (hm96.cu's dump, each binary lever against hm96.cu's tree, the
#      k-ary dumps); then hm96_ref.py check on the dumps and hm96lat_ref.py check on the k-ary dumps; lease 2, only if every
#      check passed, `bench` over every size and `overlap` over the windows with a stand-in of TARGET_MS, nvidia-smi's
#      clocks every 200 ms.
#
#   research run --on vy-dc-b200 --project verity --campaign hash --source . --cwd source --env GPU_LEASE_WHO=hash-intpipe \
#     -- bash benchmarks/hash/intpipe/hm96lat.sh
# LEASE_ARGS (default "--max-min 14"; on vy-nebius-2 "--preemptible --max-min 20"), REPS (15), WINDOWS (4614,6433,10078:
# 4 KiB leaves of the residual-folded, per-layer and per-block rows' decode windows), BATCH (2,4,8: steps a window),
# TARGET_MS (the stand-in's step: 4.08 B200, 5.65 H200, 15.0 RTX PRO 6000 by default), SKIP_LEAF (1: no step 1).
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda/bin:$PATH
export I=benchmarks/hash/intpipe PY=python3 PYTHONPATH=.
W=$(mktemp -d /dev/shm/intpipe-hm96lat.XXXXXX)
export W
trap 'rm -rf "$W"' EXIT
git rev-parse HEAD > "$O/commit.txt" 2>&1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
case $CC in 120) SM=120a ;; *) SM=$CC ;; esac
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case $GPU in *B200*) T=4.08 ;; *H200*) T=5.65 ;; *"RTX PRO 6000"*) T=15.0 ;; *) T=5.0 ;; esac
export TARGET_MS=${TARGET_MS:-$T}
echo "compute_cap $CC sm_$SM gpu $GPU target_ms $TARGET_MS" > "$O/arch.txt"
WINDOWS=${WINDOWS:-4614,6433,10078}
BATCH=${BATCH:-2,4,8}
SIZES=$WINDOWS
for k in ${BATCH//,/ }; do for w in ${WINDOWS//,/ }; do SIZES="$SIZES,$((k * w))"; done; done
export WINDOWS SIZES REPS=${REPS:-15} CHECK_N=${CHECK_N:-700}
echo "windows $WINDOWS sizes $SIZES" >> "$O/arch.txt"

if [ "${SKIP_LEAF:-0}" != 1 ]; then
  W2=${WINDOWS//,/ }
  N2=""
  for w in $W2; do N2="$N2,$((2 * w))"; done
  for v in "leaf4k 4096 1024 $SIZES" "leaf2k 2048 1024 ${N2#,}" "leaf2k-seg512 2048 512 ${N2#,}"; do
    read -r d leaf segb n <<< "$v"
    mkdir -p "$O/$d"
    OUT=$O/$d LEAF=$leaf SEGB=$segb CANDS=sha512:0:80 N=$n REPS=$REPS bash $I/hm96.sh > "$O/$d/hm96.log" 2>&1
    echo "hm96.sh $d rc $?" >> "$O/rc.txt"
  done
fi

nvcc -O3 -std=c++17 -gencode arch=compute_$SM,code=sm_$SM -Xptxas -v -DCAND_KIND=0 -DSHA_ROLL=1 -DFETCH="${FETCH:-128}" \
  -DLOADER_ASYNC="${ASYNC:-0}" -DSEG_WARPS="${WARPS:-4}" -o "$W/hm96lat" $I/hm96lat.cu -lnvidia-ml > "$O/build-hm96lat.log" 2>&1
echo "build hm96lat rc $?" >> "$O/rc.txt"
mkdir -p "$W/p"
for n in $CHECK_N ${SIZES//,/ }; do
  ( $PY $I/hm96_ref.py params "$W/p/params-$n.bin" sha512 "$n" 2>> "$O/params.err"
    echo "params $n rc $?" >> "$O/rc.txt" ) &
done
wait
cat "$O/rc.txt"
[ -x "$W/hm96lat" ] || exit 1
# shellcheck disable=SC2016
gpu-lease 1 --wait ${LEASE_ARGS:---max-min 14} -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,compute_cap,clocks.sm,clocks.max.sm,power.limit --format=csv \
    -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  for n in $CHECK_N ${SIZES//,/ }; do
    mkdir -p "$W/v-$n/base" "$W/v-$n/k4" "$W/v-$n/k8"
    "$W/hm96lat" verify "$W/p/params-$n.bin" "$n" "$W/v-$n" >> "$O/verify.jsonl" 2>> "$O/verify.err"
    echo "verify $n rc $?" >> "$O/rc.txt"
  done'
echo "lease 1 rc $?" >> "$O/rc.txt"
for n in $CHECK_N ${SIZES//,/ }; do
  ( $PY $I/hm96_ref.py check "$W/v-$n/base" > "$O/check-base-$n.json" 2> "$O/check-base-$n.err"
    echo "check base $n rc $?" >> "$O/rc.txt"
    for k in k4 k8; do
      $PY $I/hm96lat_ref.py check "$W/v-$n/$k" > "$O/check-$k-$n.json" 2> "$O/check-$k-$n.err"
      echo "check $k $n rc $?" >> "$O/rc.txt"
    done
    rm -rf "$W/v-$n" ) &
done
wait
ok=1
for n in $CHECK_N ${SIZES//,/ }; do
  for k in base k4 k8; do grep -q '"ok": true' "$O/check-$k-$n.json" 2>/dev/null || ok=0; done
done
grep -q '"ok": false' "$O/verify.jsonl" && ok=0
[ "$(grep -c '"mode": "verify"' "$O/verify.jsonl")" -gt 0 ] || ok=0
echo "checked $ok" >> "$O/rc.txt"
# shellcheck disable=SC2016
[ $ok = 1 ] && gpu-lease 1 --wait ${LEASE_ARGS:---max-min 14} -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,compute_cap,clocks.sm,clocks.max.sm,power.limit --format=csv \
    -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active \
    --format=csv -lms 200 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  "$W/hm96lat" bench "$W/p" "$REPS" "$SIZES" >> "$O/lat.jsonl" 2>> "$O/lat.err"
  echo "bench rc $?" >> "$O/rc.txt"
  "$W/hm96lat" overlap "$W/p" "$REPS" "$TARGET_MS" "$WINDOWS" >> "$O/overlap.jsonl" 2>> "$O/overlap.err"
  echo "overlap rc $?" >> "$O/rc.txt"
  kill $smi'
echo "lease 2 rc $?" >> "$O/rc.txt"
cat "$O/rc.txt"
$PY $I/summary.py lat "$O" | tee "$O/summary-lat.jsonl"
