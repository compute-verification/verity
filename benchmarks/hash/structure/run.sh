#!/usr/bin/env bash
# The structure levers on one leased GPU (cwd = the source tree's root; sm_120 or sm_90, from nvidia-smi).
# Outside a lease: builds (fuse.cu, and commit_gpu.cu for its leaf-size sweep), parameter files, commit_gpu's quick cases.
# Lease 1: fuse check (the fused and separate arms' levels equal, C equal to cuBLASLt's, dumps written) and commit_gpu's
# check. Then on the CPU: reference.py verify on every dump and check_chacha. Lease 2, only if every check passed: the
# fusion bench at Qwen3-8B's prefill and decode GEMMs, the leaf-size sweep (tile leaves R = 16 .. 128; commit_gpu rows and
# row-seg leaves 512 B .. 16 KiB), a-ary trees, and ChaCha20 salts. Clocks are read, never set.
#
#   research run --on vy-nebius-2 --project verity --campaign hash --source . --cwd source --env GPU_LEASE_WHO=hash-structure \
#     -- bash benchmarks/hash/structure/run.sh
# REPS (default 15); LEASE_MIN (default 14); FUSE_ONLY (set: lease 2 runs the fusion bench only); PREFILL_M (default 4096); SHAPES / DECODE (N:K lists); LEASE_ARGS (extra gpu-lease flags)
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda/bin:/usr/local/cuda-13.0/bin:$PATH
export S=benchmarks/hash/structure CG=benchmarks/pouw/commit_gpu PY=python3 PYTHONPATH=.
W=$(mktemp -d /dev/shm/hash-structure.XXXXXX)
export W
trap 'rm -rf "$W"' EXIT
git rev-parse HEAD > "$O/commit.txt" 2>&1
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader > "$O/host-gpus.txt" 2>&1
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
case $cap in 120) arch=compute_120a,code=sm_120a ;; 90) arch=compute_90a,code=sm_90a ;; *) arch=compute_${cap},code=sm_${cap} ;; esac
echo "arch $arch" >> "$O/rc.txt"
nvcc -O3 -std=c++17 -gencode arch=$arch -Xptxas -v -I$CG -o "$W/fuse" $S/fuse.cu -lcublasLt -lnvidia-ml > "$O/build-fuse.log" 2>&1 &
nvcc -O3 -std=c++17 -gencode arch=$arch -Xptxas -v -DSHA_ROLL=1 -o "$W/commit_gpu" $CG/commit_gpu.cu -lcublasLt -lnvidia-ml \
  > "$O/build-commit_gpu.log" 2>&1 &
$PY $CG/reference.py prepare "$W/cases" quick > "$O/prepare.jsonl" 2> "$O/prepare.err" &
export M=${PREFILL_M:-4096}
# Qwen3-8B (hidden 4096, intermediate 12288, 32 q heads and 8 kv heads of 128): qkv, o, gate_up, down as N:K
export SHAPES=${SHAPES:-6144:4096 4096:4096 24576:4096 4096:12288} DECODE=${DECODE:-6144:4096 4096:4096 24576:4096 4096:12288}
params() {   # params M N R -> its file
  local f="$W/p-$1-$2-$3.bin" n=$(( ($1 + $3 - 1) / $3 * (2 * $2 / 32) ))
  [ -e "$f" ] || $PY $CG/reference.py params "$f" "$n" $((256 * $3)) --ks-seed 7
  echo "$f"
}
# the check cases (M N K R): ragged M, decode, o_proj's full prefill shape
CHECKS="256 256 256 16|200 384 128 64|32 512 256 32|4096 4096 4096 64|4096 6144 4096 128"
IFS='|'; for c in $CHECKS; do IFS=' '; set -- $c; params "$1" "$2" "$4" > /dev/null; done; IFS=' '
for nk in $SHAPES; do params "$M" "${nk%%:*}" 64 > /dev/null; done
for nk in 4096:4096 24576:4096; do for R in 16 32 128; do params "$M" "${nk%%:*}" $R > /dev/null; done; done
for nk in $DECODE; do for R in 16 32; do params 32 "${nk%%:*}" $R > /dev/null; done; done
export SWEEP=${SWEEP:-512 1024 2048 4096} SEG_SWEEP=${SEG_SWEEP:-1024 2048 4096 8192 16384}
for L in $SWEEP; do $PY $CG/reference.py params "$W/r-rows-$L.bin" $((33554432 / L)) $((8 * L)); done
for L in $SEG_SWEEP; do $PY $CG/reference.py params "$W/r-seg-$L.bin" $((33554432 / L)) $((8 * L)) --seg; done
$PY $CG/reference.py params "$W/p-tree.bin" 1048576 256
$PY $CG/reference.py params "$W/p-chacha.bin" 1048576 256
wait
grep -E "registers|spill" "$O/build-fuse.log" | sed 's/^ptxas info *: //' > "$O/regs-fuse.txt"
[ -x "$W/fuse" ] && echo "build fuse ok" >> "$O/rc.txt" || { echo "build fuse FAILED" >> "$O/rc.txt"; cat "$O/rc.txt"; exit 1; }

export REPS=${REPS:-15} LEASE_MIN=${LEASE_MIN:-14} CHECKS
# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "$LEASE_MIN" ${LEASE_ARGS:-} -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,power.limit --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  i=0; IFS="|"
  for c in $CHECKS; do
    IFS=" "; set -- $c
    d="$W/dump$i"; mkdir -p "$d"
    "$W/fuse" check "$W/p-$1-$2-$4.bin" "$d" "$1" "$2" "$3" "$4" >> "$O/check-fuse.jsonl" 2>> "$O/check-fuse.err"
    echo "fuse check $c rc $?" >> "$O/rc.txt"
    i=$((i + 1)); IFS="|"
  done
  IFS=" "
  "$W/commit_gpu" selftest >> "$O/check-commit_gpu.jsonl"
  $PY $CG/reference.py check "$W/commit_gpu" "$W/cases" cg >> "$O/check-commit_gpu.jsonl" 2> "$O/check-commit_gpu.err"
  echo "commit_gpu check rc $?" >> "$O/rc.txt"
  "$W/fuse" chacha "$W/p-chacha.bin" "$O" 4 1 > /dev/null 2>> "$O/check-fuse.err"
  echo "chacha dump rc $?" >> "$O/rc.txt"'
echo "lease 1 rc $?" >> "$O/rc.txt"
$PY $CG/reference.py verify "$W"/dump* > "$O/verify.jsonl" 2> "$O/verify.err"; v=$?
echo "reference verify rc $v" >> "$O/rc.txt"
$PY $S/check_chacha.py "$O" > "$O/check-chacha.json"; ch=$?
echo "check_chacha rc $ch" >> "$O/rc.txt"
fc=$(grep -c "\"check\": true" "$O/check-fuse.jsonl"); nc=$(echo "$CHECKS" | tr "|" "\n" | wc -l)
cg=$(tail -1 "$O/check-commit_gpu.jsonl" | grep -c "\"failed\": 0")
echo "fuse checks passed $fc of $nc; commit_gpu quick cases pass $cg" >> "$O/rc.txt"
if [ "$v" != 0 ] || [ "$ch" != 0 ] || [ "$fc" != "$nc" ] || [ "$cg" != 1 ]; then
  echo "a check failed: nothing is timed" >> "$O/rc.txt"; cat "$O/rc.txt"; exit 1
fi

# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "$LEASE_MIN" ${LEASE_ARGS:-} -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,power.limit --format=csv -i "$GPU_LEASE_UUID" >> "$O/gpu.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu --format=csv -lms 500 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  b() { "$W/fuse" bench "$W/p-$1-$2-$4.bin" "$1" "$2" "$3" "$4" "$REPS" >> "$O/bench-fuse.jsonl" 2>> "$O/bench.err"; echo "bench $* rc $?" >> "$O/rc.txt"; }
  for nk in $SHAPES; do b "$M" "${nk%%:*}" "${nk##*:}" 64; done
  for nk in $DECODE; do for R in 32 16; do b 32 "${nk%%:*}" "${nk##*:}" $R; done; done
  for nk in 4096:4096 24576:4096; do for R in 16 32 128; do b "$M" "${nk%%:*}" "${nk##*:}" $R; done; done
  [ -n "${FUSE_ONLY:-}" ] && { kill $smi; exit 0; }
  for n in 16384 131072 1048576; do "$W/fuse" tree "$W/p-tree.bin" $n "$REPS" >> "$O/tree.jsonl" 2>> "$O/bench.err"; done
  echo "tree rc $?" >> "$O/rc.txt"
  for n in 98304 1048576; do "$W/fuse" chacha "$W/p-chacha.bin" "$W" $n "$REPS" >> "$O/chacha.jsonl" 2>> "$O/bench.err"; done
  echo "chacha rc $?" >> "$O/rc.txt"
  "$W/commit_gpu" sha 10 > "$O/sha.jsonl" 2>> "$O/bench.err"
  for L in $SWEEP; do
    "$W/commit_gpu" bench "$W/r-rows-$L.bin" rows $((33554432 / L)) $((8 * L)) 9 8192 4096 4096 >> "$O/bench-leaf.jsonl" 2>> "$O/bench.err"
    echo "leaf rows $L rc $?" >> "$O/rc.txt"
  done
  for L in $SEG_SWEEP; do
    "$W/commit_gpu" bench "$W/r-seg-$L.bin" seg $((33554432 / L)) $((8 * L)) 9 8192 4096 4096 >> "$O/bench-leaf.jsonl" 2>> "$O/bench.err"
    echo "leaf seg $L rc $?" >> "$O/rc.txt"
  done
  kill $smi'
echo "lease 2 rc $?" >> "$O/rc.txt"
cat "$O/rc.txt"
$PY $S/summary.py "$O" > "$O/summary.md" 2> "$O/summary.err"
cat "$O/summary.md"
