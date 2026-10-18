#!/usr/bin/env bash
# Red-team re-run of intpipe's decode-window overlap lever on one leased GPU (cwd = this branch's root): intpipe's tree fetched
# at its pushed commit, decode_overlap.cu built on its hm96lat.cu with hm96lat.sh's flags. Lease 1: hm96lat's `verify` at
# CHECK_N and each window (its dump of the base tree, every binary lever against it); then hm96_ref.py check on each dump
# (verity.ml.commitments); only if all pass, lease 2: `overlap`, the base commit beside a Qwen3-8B decode step of cuBLASLt
# GEMMs and beside hm96lat's stand-in (at the step's time and at TARGET_MS), in one process.
#
#   research run --on vy-dc-h200 --project verity --campaign hash-loop --source . --cwd source \
#     --env GPU_LEASE_WHO=kernels-redteam -- bash benchmarks/hash/redteam/decode.sh
# INTPIPE_COMMIT (default: cursor/hash-intpipe-2550's head 4dc1c9787), REPS (15), WINDOWS (4614,6433,10078), TARGET_MS
# (4.08 B200, 5.65 H200, 15.0 RTX PRO 6000: the claim's), B (32), KV_MB (48), LEASE_ARGS (--preemptible on vy-nebius-2)
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda/bin:$PATH
R=$(pwd)/benchmarks/hash/redteam
SHA=${INTPIPE_COMMIT:-4dc1c978709d0f8ecd1633564bf09820324bce47}
W=$(mktemp -d)
export W
trap 'rm -rf "$W"' EXIT
git rev-parse HEAD > "$O/probe_commit.txt"
S="$W/src"
mkdir -p "$S" && git -C "$S" init -q && git -C "$S" remote add origin https://github.com/compute-verification/verity.git
if ! git -C "$S" fetch -q --depth=1 --filter=blob:none origin "$SHA" 2> "$O/fetch.err"; then
  git -C "$S" fetch -q --depth=1 origin "$SHA" 2>> "$O/fetch.err" || { echo "fetch failed" >> "$O/rc.txt"; exit 1; }
fi
git -C "$S" sparse-checkout set --no-cone /benchmarks/hash/intpipe /benchmarks/pouw/commit_gpu /archive/pouw-blake3/pearl_c /verity >> "$O/fetch.err" 2>&1
git -C "$S" checkout -q FETCH_HEAD 2>> "$O/fetch.err"
[ "$(git -C "$S" rev-parse HEAD)" = "$SHA" ] || { echo "checkout is not $SHA" >> "$O/rc.txt"; exit 1; }
echo "$SHA" > "$O/intpipe_commit.txt"
I="$S/benchmarks/hash/intpipe"
sha256sum "$I/hm96.cu" "$I/hm96lat.cu" "$I/hashes.cuh" > "$O/sources.sha256"
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | head -1 | tr -d .)
case $CC in 120) SM=120a ;; *) SM=$CC ;; esac
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | head -1)
case $GPU in *B200*) T=4.08 ;; *H200*) T=5.65 ;; *"RTX PRO 6000"*) T=15.0 ;; *) T=5.0 ;; esac
export TARGET_MS=${TARGET_MS:-$T} WINDOWS=${WINDOWS:-4614,6433,10078} REPS=${REPS:-15} CHECK_N=${CHECK_N:-700}
echo "arch $CC sm_$SM gpu $GPU target_ms $TARGET_MS windows $WINDOWS" > "$O/rc.txt"
nvcc -O3 -std=c++17 -gencode arch=compute_$SM,code=sm_$SM -Xptxas -v -DCAND_KIND=0 -DSHA_ROLL=1 -DFETCH=128 -DLOADER_ASYNC=0 \
  -DSEG_WARPS=4 -DHM96LAT_CU="\"$I/hm96lat.cu\"" -o "$W/dec" "$R/decode_overlap.cu" -lnvidia-ml -lcublasLt > "$O/build.log" 2>&1
echo "build rc $?" >> "$O/rc.txt"
mkdir -p "$W/p"
NS="$CHECK_N ${WINDOWS//,/ }"
for n in $NS; do
  (cd "$S" && PYTHONPATH=. python3 benchmarks/hash/intpipe/hm96_ref.py params "$W/p/params-$n.bin" sha512 "$n") 2>> "$O/params.err" ||
    echo "params $n failed" >> "$O/rc.txt"
done
cat "$O/rc.txt"
[ -x "$W/dec" ] || { tail -40 "$O/build.log"; exit 1; }
export NS
# shellcheck disable=SC2016,SC2086
gpu-lease 1 --wait --max-min 8 ${LEASE_ARGS:-} -- bash -c '
  for n in $NS; do
    mkdir -p "$W/v-$n/base" "$W/v-$n/k4" "$W/v-$n/k8"
    "$W/dec" verify "$W/p/params-$n.bin" "$n" "$W/v-$n" >> "$O/verify.jsonl" 2>> "$O/verify.err"
    echo "verify $n rc $?" >> "$O/rc.txt"
  done'
ok=1
for n in $NS; do
  (cd "$S" && PYTHONPATH=. python3 benchmarks/hash/intpipe/hm96_ref.py check "$W/v-$n/base") > "$O/check-$n.json" 2> "$O/check-$n.err"
  grep -q '"ok": true' "$O/check-$n.json" || ok=0
done
grep -q '"ok": false' "$O/verify.jsonl" && ok=0
[ "$(grep -c '"mode": "verify"' "$O/verify.jsonl")" -gt 0 ] || ok=0
echo "checked $ok" >> "$O/rc.txt"
[ $ok = 1 ] || { cat "$O/rc.txt"; exit 1; }
# shellcheck disable=SC2016,SC2086
gpu-lease 1 --wait --max-min 12 ${LEASE_ARGS:-} -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,power.limit,memory.used --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active --format=csv -lms 200 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  "$W/dec" overlap "$W/p" "$REPS" "$TARGET_MS" "$WINDOWS" > "$O/overlap.jsonl" 2> "$O/overlap.err"
  echo "overlap rc $?" >> "$O/rc.txt"
  kill $smi'
echo "lease rc $?" >> "$O/rc.txt"
cat "$O/rc.txt"
grep -h -E '"tuned"|"decode_step"|"standin"|"summary"|decode_overlap_fails' "$O/overlap.jsonl"
