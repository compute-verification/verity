#!/usr/bin/env bash
# overlap on one leased GPU (cwd = the source tree's root). Outside a lease: the build (GEMMS=none, or sm90 / sm100 with
# CUTLASS fetched at CUTLASS_TAG into scratch) and reference.py's parameter files. Lease 1: `overlap dump`, every hash
# variant's levels beside a running GEMM, device-compared with commit_gpu's Committer. Then reference.py verify on every
# dump (verity.ml.commitments), and only if every dump verifies, lease 2: `overlap bench`. PROF=1: lease 3, an nsys trace
# of every GEMM and hash kernel (grid, block, registers, shared memory). Clocks are read, never set.
#
#   research run --on vy-dc-b200 --project verity --campaign hash --source . --cwd source \
#     --env GPU_LEASE_WHO=hash-prefill-overlap -- bash benchmarks/hash/overlap/gpu.sh
# REPS (default 9); GEMMS; LEASE_ARGS (e.g. --preemptible on vy-nebius-2); PROF (default 1)
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda/bin:$PATH PYTHONPATH=.
D=benchmarks/hash/overlap
W=$(mktemp -d)
export W
trap 'rm -rf "$W"' EXIT
git rev-parse HEAD > "$O/commit.txt" 2>&1
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | head -1 | tr -d .)
GEN="-gencode arch=compute_${cc}a,code=sm_${cc}a"
GEMMS=${GEMMS:-none}
CUTLASS_TAG=${CUTLASS_TAG:-v4.2.1}
echo "arch $cc gemms $GEMMS" > "$O/rc.txt"
INC=""
if [ "$GEMMS" != none ]; then
  git clone -q --depth 1 --branch "$CUTLASS_TAG" https://github.com/NVIDIA/cutlass.git "$W/cutlass" 2> "$O/cutlass-clone.err"
  git -C "$W/cutlass" rev-parse HEAD > "$O/cutlass.txt"
  bash $D/patch_cutlass.sh "$W/cutlass" >> "$O/cutlass.txt" 2>&1
  INC="-I$W/cutlass/include -I$W/cutlass/tools/util/include --expt-relaxed-constexpr"
fi
mkdir -p "$W/p"
for n in 5120 7680 11520 15360; do
  python3 benchmarks/pouw/commit_gpu/reference.py params "$W/p/p-$n.bin" $n 32768 --ks-seed 11 || echo "params $n failed" >> "$O/rc.txt"
done
# shellcheck disable=SC2086
nvcc -O3 -std=c++17 $GEN -Xptxas -v $INC -c -o "$W/gemms.o" $D/gemms_$GEMMS.cu ${GEMM_FLAGS:-} > "$O/build-gemms.log" 2>&1
echo "build gemms rc $?" >> "$O/rc.txt"
# shellcheck disable=SC2086
nvcc -O3 -std=c++17 $GEN -Xptxas -v -o "$W/overlap" $D/overlap.cu "$W/gemms.o" -lcublasLt -lnvidia-ml $INC > "$O/build.log" 2>&1
echo "build rc $?" >> "$O/rc.txt"
cat "$O/rc.txt"
[ -x "$W/overlap" ] || { tail -30 "$O/build-gemms.log" "$O/build.log"; exit 1; }
# shellcheck disable=SC2016,SC2086
gpu-lease 1 --wait --max-min 10 ${LEASE_ARGS:-} -- bash -c '"$W/overlap" dump "$W/p" "$W/dumps" > "$O/dump.jsonl" 2> "$O/dump.err"; echo "dump rc $?" >> "$O/rc.txt"'
python3 benchmarks/pouw/commit_gpu/reference.py verify "$W"/dumps/*-d* > "$O/verify.jsonl" 2> "$O/verify.err"
v=$?
echo "verify rc $v" >> "$O/rc.txt"
tail -1 "$O/verify.jsonl"
if [ $v != 0 ] || ! grep -q '"failed": 0' "$O/verify.jsonl"; then
  echo "reference verify failed: no timing" >> "$O/rc.txt"
  cat "$O/rc.txt"
  exit 1
fi
export REPS=${REPS:-9}
# shellcheck disable=SC2016,SC2086
gpu-lease 1 --wait --max-min 25 ${LEASE_ARGS:-} -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,power.limit,memory.used --format=csv -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu --format=csv -lms 200 -i "$GPU_LEASE_UUID" > "$O/smi.csv" 2>&1 &
  smi=$!
  "$W/overlap" bench "$W/p" $REPS > "$O/bench.jsonl" 2> "$O/bench.err"
  echo "bench rc $?" >> "$O/rc.txt"
  kill $smi'
if [ "${PROF:-1}" = 1 ]; then
  # shellcheck disable=SC2016,SC2086
  gpu-lease 1 --wait --max-min 10 ${LEASE_ARGS:-} -- bash -c '
    nsys profile -t cuda -o "$W/t" --force-overwrite true "$W/overlap" nsys "$W/p" > "$O/nsys-run.jsonl" 2> "$O/nsys.err"
    nsys stats -r cuda_gpu_trace -f csv -o "$O/nsys" "$W/t.nsys-rep" >> "$O/nsys.err" 2>&1
    echo "nsys rc $?" >> "$O/rc.txt"'
fi
python3 $D/summary.py "$O" > "$O/summary.md" 2> "$O/summary.err"
cat "$O/rc.txt"
cat "$O/summary.md"
