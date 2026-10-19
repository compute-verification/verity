#!/usr/bin/env bash
# session.cu before and after libvc's device functions (`../lib/vc_device.cuh`) on one leased GPU (cwd = the source tree):
# both built for this node's GPU outside the lease (the before's text shipped as inputs/session_base.cu, origin/main's),
# then under one lease memory accounting's pass gate on the after (`inplace_gate.py --no-timing`: every receipt, root,
# kept level and opening against the driver's reference), the in-place pass of both interleaved at 8B (`ab_pass.py`), and
# the served pass (`pass_bench.py`: rs_pass and the chained passes, each root against reader.cu's unfused tree) after,
# before, after. A gate that fails ends the lease before anything is timed. Results go to $OUT.
#
#   git show origin/main:benchmarks/remote_session/session.cu > /tmp/session_base.cu
#   research run --on vy-dc-h200 --project verity --campaign hash-loop --source . --cwd source --timeout 60m \
#     --send /tmp/session_base.cu --env GPU_LEASE_WHO=kernels-bwkernels -- bash benchmarks/hash/bwkernels/ab.sh
# LEASE_ARGS (--preemptible on vy-nebius-2), MAXMIN (the lease, default 40), STEPS (default "gate ab pb"), ROUNDS, REPS.
set -uo pipefail
export O=${OUT:?}
mkdir -p "$O"
IN=${RESEARCH_RUN_DIR:?}/inputs
RS=benchmarks/remote_session
export RS
PY=${PY:-}
for p in /workspace/pouw/fill-out/pous/work/pous-bench/gpu312-torch2.14.0-cu130/bin/python \
         /workspace/pouw/mvp-e2e/venv312/bin/python /workspace/cp/venv312/bin/python; do
  [ -z "$PY" ] && [ -x "$p" ] && PY=$p
done
[ -n "$PY" ] || { echo "no torch venv on this node"; exit 3; }
CUDA_HOME=${CUDA_HOME:-}
for c in /usr/local/cuda-13.0 /usr/local/cuda; do
  [ -z "$CUDA_HOME" ] && [ -x "$c/bin/nvcc" ] && CUDA_HOME=$c
done
export CUDA_HOME PY PATH=$CUDA_HOME/bin:$(dirname "$PY"):$PATH
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
B=$(mktemp -d /tmp/bwkernels.XXXXXX)
export B
trap 'rm -rf "$B"' EXIT
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | head -1 | tr -d .)
{ nvcc --version; echo "python: $PY"; echo "sm_${CC}a"; nvidia-smi --query-gpu=name,driver_version --format=csv -i 0; } \
  > "$O/build.txt" 2>&1
FLAGS=(-O3 -gencode "arch=compute_${CC}a,code=sm_${CC}a" -std=c++17 -Xcompiler -fPIC -shared -Xptxas -v -I "$RS")
nvcc "${FLAGS[@]}" -o "$B/after.so" $RS/session.cu > "$O/after.ptxas" 2>&1 & A=$!
nvcc "${FLAGS[@]}" -o "$B/before.so" "$IN/session_base.cu" > "$O/before.ptxas" 2>&1 & Z=$!
wait $A; ra=$?
wait $Z; rz=$?
echo "build after rc $ra before rc $rz" | tee "$O/rc.txt"
for k in after before; do
  grep -E "Compiling entry|registers|spill" "$O/$k.ptxas" > "$O/$k.regs.txt" 2>/dev/null
done
[ "$ra" = 0 ] && [ "$rz" = 0 ] || { tail -30 "$O/after.ptxas" "$O/before.ptxas"; exit 4; }
export STEPS=${STEPS:-gate ab pb} ROUNDS=${ROUNDS:-12} REPS=${REPS:-5}
# shellcheck disable=SC2016,SC2086
gpu-lease 1 --wait --max-min "${MAXMIN:-40}" ${LEASE_ARGS:-} -- bash -c '
  nvidia-smi --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,power.limit --format=csv \
    -i "$GPU_LEASE_UUID" > "$O/gpu.txt" 2>&1
  for s in $STEPS; do
    case $s in
      gate) "$PY" $RS/inplace_gate.py --lib "$B/after.so" --out "$O/gate" --no-timing > "$O/gate.log" 2>&1
            rc=$?; echo "gate rc $rc" >> "$O/rc.txt"; [ $rc = 0 ] || exit 1 ;;
      ab) "$PY" benchmarks/hash/bwkernels/ab_pass.py --before "$B/before.so" --after "$B/after.so" --out "$O" \
            --rounds "$ROUNDS" --reps "$REPS" > "$O/ab.log" 2>&1
          echo "ab rc $?" >> "$O/rc.txt" ;;
      pb) i=0
          for k in after before after; do
            i=$((i + 1))
            "$PY" $RS/pass_bench.py --lib "$B/$k.so" --gib 16 --chunks 16 --passes 20 --open-passes 20 \
              --openings 2000 --contend "" --out "$O/pb_${i}_$k" > "$O/pb_${i}_$k.log" 2>&1
            echo "pb $i $k rc $?" >> "$O/rc.txt"
          done ;;
    esac
  done'
echo "lease rc $?" >> "$O/rc.txt"
cat "$O/rc.txt"
tail -2 "$O/gate.log" 2>/dev/null
tail -1 "$O/ab.log" 2>/dev/null
grep -h '"root_ok"' "$O"/pb_*.log 2>/dev/null | cut -c1-400
