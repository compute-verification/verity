#!/usr/bin/env bash
# Red-team re-run of prefill-overlap's B200 claim on one leased GPU (cwd = this branch's root): its tree fetched at its pushed
# commit and built as its gpu.sh builds it (CUTLASS at its tag, patched, the same four GEMM configurations), plus
# overlap_gemm.cu (its overlap.cu unedited, with a word-for-word GEMM output check). Lease 1: `overlap dump`; then
# reference.py verify on every dump (verity.ml.commitments); only if all verify, lease 2: the GEMM check, then `overlap bench`
# (its gates, then the timed table) as a second sample of the published one.
#
#   research run --on vy-dc-b200 --project verity --campaign hash-loop --source . --cwd source \
#     --env GPU_LEASE_WHO=kernels-redteam -- bash benchmarks/hash/redteam/overlap.sh
# OVERLAP_COMMIT (default: cursor/hash-prefill-overlap-8d50's head 32c1ee714), REPS (9), LEASE_ARGS
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda/bin:$PATH
R=$(pwd)/benchmarks/hash/redteam
SHA=${OVERLAP_COMMIT:-32c1ee714910f145a5bceb66fdf8d9ca7522f0ed}
CFGS="u1_128x256:100:0:128:256:64:1:1:0:0:0:0 u2_256x256:100:1:256:256:64:2:1:0:0:0:0 u1_128x128:100:0:128:128:64:1:1:0:0:0:0 u2_256x128:100:1:256:128:64:2:1:0:0:0:0"
W=$(mktemp -d)
export W
trap 'rm -rf "$W"' EXIT
git rev-parse HEAD > "$O/probe_commit.txt"
S="$W/src"
mkdir -p "$S" && git -C "$S" init -q && git -C "$S" remote add origin https://github.com/compute-verification/verity.git
if ! git -C "$S" fetch -q --depth=1 --filter=blob:none origin "$SHA" 2> "$O/fetch.err"; then
  git -C "$S" fetch -q --depth=1 origin "$SHA" 2>> "$O/fetch.err" || { echo "fetch failed" >> "$O/rc.txt"; exit 1; }
fi
git -C "$S" sparse-checkout set --no-cone /benchmarks/hash/overlap /benchmarks/pouw/commit_gpu /verity >> "$O/fetch.err" 2>&1
git -C "$S" checkout -q FETCH_HEAD 2>> "$O/fetch.err"
[ "$(git -C "$S" rev-parse HEAD)" = "$SHA" ] || { echo "checkout is not $SHA" >> "$O/rc.txt"; exit 1; }
echo "$SHA" > "$O/overlap_commit.txt"
D="$S/benchmarks/hash/overlap"
sha256sum "$D/overlap.cu" "$D/gemm_cutlass.cu" "$D/gemms.cu" > "$O/sources.sha256"
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | head -1 | tr -d .)
GEN="-gencode arch=compute_${cc}a,code=sm_${cc}a"
echo "arch $cc" > "$O/rc.txt"
git clone -q --depth 1 --branch "${CUTLASS_TAG:-v4.2.1}" https://github.com/NVIDIA/cutlass.git "$W/cutlass" 2> "$O/cutlass-clone.err"
git -C "$W/cutlass" rev-parse HEAD > "$O/cutlass.txt"
bash "$D/patch_cutlass.sh" "$W/cutlass" >> "$O/cutlass.txt" 2>&1
INC="-I$W/cutlass/include -I$W/cutlass/tools/util/include --expt-relaxed-constexpr"
mkdir -p "$W/p"
(cd "$S" && for n in 5120 7680 11520 15360; do
  PYTHONPATH=. python3 benchmarks/pouw/commit_gpu/reference.py params "$W/p/p-$n.bin" $n 32768 --ks-seed 11 || echo "params $n failed" >> "$O/rc.txt"
done) 2> "$O/params.err"
nvcc -O3 -std=c++17 $GEN -c -o "$W/gemms.o" "$D/gemms.cu" > "$O/build-gemms.log" 2>&1
objs=("$W/gemms.o")
for cfg in $CFGS; do
  IFS=: read -r nm ar kd tm tn tk cm cn sg mx lr mr <<< "$cfg"
  defs=(-DOV_NAME="\"$nm\"" -DOV_ID="$nm" -DOV_ARCH="$ar" -DOV_KIND="$kd" -DOV_TM="$tm" -DOV_TN="$tn" -DOV_TK="$tk" -DOV_CM="$cm" -DOV_CN="$cn" -DOV_STAGES="$sg")
  # shellcheck disable=SC2086
  (nvcc -O3 -std=c++17 $GEN -Xptxas -v $INC "${defs[@]}" -c -o "$W/g-$nm.o" "$D/gemm_cutlass.cu" > "$O/build-g-$nm.log" 2>&1
   echo "build $nm rc $?" >> "$O/rc.txt") &
done
wait
for cfg in $CFGS; do nm=${cfg%%:*}; [ -f "$W/g-$nm.o" ] && objs+=("$W/g-$nm.o"); done
# shellcheck disable=SC2086
nvcc -O3 -std=c++17 $GEN -Xptxas -v -o "$W/overlap" "$D/overlap.cu" "${objs[@]}" -lcublasLt -lnvidia-ml $INC > "$O/build.log" 2>&1
echo "build overlap rc $?" >> "$O/rc.txt"
# shellcheck disable=SC2086
nvcc -O3 -std=c++17 $GEN -DOVERLAP_CU="\"$D/overlap.cu\"" -o "$W/probe" "$R/overlap_gemm.cu" "${objs[@]}" -lcublasLt -lnvidia-ml $INC > "$O/build-probe.log" 2>&1
echo "build probe rc $?" >> "$O/rc.txt"
cat "$O/rc.txt"
[ -x "$W/overlap" ] && [ -x "$W/probe" ] || { tail -30 "$O/build.log" "$O/build-probe.log"; exit 1; }
# shellcheck disable=SC2016,SC2086
gpu-lease 1 --wait --max-min 10 ${LEASE_ARGS:-} -- bash -c '"$W/overlap" dump "$W/p" "$W/dumps" > "$O/dump.jsonl" 2> "$O/dump.err"; echo "dump rc $?" >> "$O/rc.txt"'
(cd "$S" && PYTHONPATH=. python3 benchmarks/pouw/commit_gpu/reference.py verify "$W"/dumps/*-*) > "$O/verify.jsonl" 2> "$O/verify.err"
v=$?
echo "verify rc $v" >> "$O/rc.txt"
if [ $v != 0 ] || ! grep -q '"failed": 0' "$O/verify.jsonl"; then
  echo "reference verify failed: no timing" >> "$O/rc.txt"; cat "$O/rc.txt"; exit 1
fi
export REPS=${REPS:-9}
# shellcheck disable=SC2016,SC2086
gpu-lease 1 --wait --max-min 15 ${LEASE_ARGS:-} -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,power.limit,memory.used --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active --format=csv -lms 200 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  "$W/probe" gemm "$W/p" > "$O/gemm.jsonl" 2> "$O/gemm.err"; echo "gemm probe rc $?" >> "$O/rc.txt"
  "$W/overlap" bench "$W/p" $REPS > "$O/bench.jsonl" 2> "$O/bench.err"; echo "bench rc $?" >> "$O/rc.txt"
  kill $smi'
echo "lease rc $?" >> "$O/rc.txt"
cat "$O/rc.txt"
grep -h -E '"gemm_bits"|"gemm_ref"|gemm_probe_fails' "$O/gemm.jsonl"
tail -1 "$O/verify.jsonl"
grep -h -E '"chain_summary"' "$O/bench.jsonl" | cut -c1-260
