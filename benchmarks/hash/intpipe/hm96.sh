#!/usr/bin/env bash
# hm96 leaf mode (hm96.cu): base SHA-512 against each candidate in Verity's leaf format, on one leased GPU of whatever arch
# the machine has. cwd = the source tree's root. Outside the lease: one binary a candidate (CANDS), each size's parameter
# file (hm96_ref.py params; CHECK_N and every N). Lease 1: each binary's dump at CHECK_N and at every N. Then hm96_ref.py
# check on every dump (CPU: the whole tree at CHECK_N, sampled leaves, pads and nodes of every level at each N). Lease 2:
# every binary whose dumps all passed, its bench (REPS reps over each N, base and candidate back to back), with
# nvidia-smi's clocks every 200 ms.
#
#   research run --on vy-dc-h200 --project verity --campaign hash --source . --cwd source --env GPU_LEASE_WHO=hash-intpipe \
#     -- bash benchmarks/hash/intpipe/hm96.sh
# PHASE (all; prep: the builds and parameter files only; gpu: the rest, from HM96_W's), HM96_W (the work dir, kept),
# LEASE_ARGS (default "--max-min 14"; on vy-nebius-2 "--preemptible --max-min 20"; inside a caller's lease none is taken),
# REPS (15), N (34668,262144: 142 MB and 1 GiB), CHECK_N (700), FETCH (128), ASYNC (0), WARPS (4: warps a segment CTA), LEAF (4096) and SEGB (1024): the leaf's
# bytes and its segments' (Verity's row-seg/v1 fixes 1024-byte segments; any other SEGB is a FORMAT CHANGE).
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda/bin:$PATH
export I=benchmarks/hash/intpipe PY=python3 PYTHONPATH=.
if [ -n "${HM96_W:-}" ]; then
  W=$HM96_W
  mkdir -p "$W"
else
  W=$(mktemp -d /dev/shm/intpipe-hm96.XXXXXX)
  trap 'rm -rf "$W"' EXIT
fi
export W
# a lease of its own, or the caller's when it already holds one
lease() {
  if [ -n "${GPU_LEASE_UUID:-}" ]; then bash -c "$1"; else gpu-lease 1 --wait ${LEASE_ARGS:---max-min 14} -- bash -c "$1"; fi
}
git rev-parse HEAD > "$O/commit.txt" 2>&1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
case $CC in 120) SM=120a ;; *) SM=$CC ;; esac
echo "compute_cap $CC sm_$SM" > "$O/arch.txt"
# name:kind:rounds
CANDS=${CANDS:-sha512:0:80 sha512r32:1:32 sha512r24:1:24 blake3k:3:0}
export HM96_LEAF=${LEAF:-4096} HM96_SEG=${SEGB:-1024}
export N=${N:-34668,262144} CHECK_N=${CHECK_N:-700} REPS=${REPS:-15}
SIZES="$CHECK_N ${N//,/ }"
export SIZES
PHASE=${PHASE:-all}
[ "$PHASE" != gpu ] && for c in $CANDS; do
  IFS=: read -r name kind r <<< "$c"
  ( nvcc -O3 -std=c++17 -gencode arch=compute_$SM,code=sm_$SM -Xptxas -v -DCAND_KIND="$kind" -DCAND_R="$r" -DSHA_ROLL=1 \
      -DFETCH="${FETCH:-128}" -DLOADER_ASYNC="${ASYNC:-0}" -DSEG_WARPS="${WARPS:-4}" \
      -DLEAF_BYTES="$HM96_LEAF" -DSEG_BYTES="$HM96_SEG" \
      -o "$W/hm96-$name" $I/hm96.cu -lnvidia-ml > "$O/build-$name.log" 2>&1
    echo "build $name rc $?" >> "$O/rc.txt" ) &
  mkdir -p "$W/p-$name"
  for n in $SIZES; do
    ( $PY $I/hm96_ref.py params "$W/p-$name/params-$n.bin" "$name" "$n" 2>> "$O/params.err"
      echo "params $name $n rc $?" >> "$O/rc.txt" ) &
  done
done
wait
[ "$PHASE" = prep ] && exit 0
BINS=""
for c in $CANDS; do
  IFS=: read -r name _ <<< "$c"
  [ -x "$W/hm96-$name" ] && BINS="$BINS $name"
done
export BINS
cat "$O/rc.txt"
# shellcheck disable=SC2086,SC2016
lease '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,compute_cap,clocks.sm,clocks.max.sm,power.limit --format=csv \
    -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  for b in $BINS; do
    for n in $SIZES; do
      mkdir -p "$W/dump-$b-$n"
      "$W/hm96-$b" dump "$W/p-$b/params-$n.bin" "$W/dump-$b-$n" "$n" 2>> "$O/dump.err"
      echo "dump $b $n rc $?" >> "$O/rc.txt"
    done
  done'
echo "lease 1 rc $?" >> "$O/rc.txt"
for b in $BINS; do
  for n in $SIZES; do
    ( $PY $I/hm96_ref.py check "$W/dump-$b-$n" > "$O/check-$b-$n.json" 2> "$O/check-$b-$n.err"
      echo "check $b $n rc $?" >> "$O/rc.txt"
      rm -rf "$W/dump-$b-$n" ) &
  done
done
wait
CHECKED=""
for b in $BINS; do
  ok=1
  for n in $SIZES; do grep -q '"ok": true}$' "$O/check-$b-$n.json" 2>/dev/null || ok=0; done
  [ $ok = 1 ] && CHECKED="$CHECKED $b"
done
echo "checked:$CHECKED" >> "$O/rc.txt"
export CHECKED
# shellcheck disable=SC2086,SC2016
[ -n "$CHECKED" ] && lease '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,compute_cap,clocks.sm,clocks.max.sm,power.limit --format=csv \
    -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active \
    --format=csv -lms 200 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  for b in $CHECKED; do
    "$W/hm96-$b" bench "$W/p-$b" "$REPS" "$N" >> "$O/bench.jsonl" 2>> "$O/bench.err"
    echo "bench $b rc $?" >> "$O/rc.txt"
  done
  kill $smi'
echo "lease 2 rc $?" >> "$O/rc.txt"
cat "$O/rc.txt"
$PY $I/summary.py hm96 "$O" | tee "$O/summary.jsonl"
