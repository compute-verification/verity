#!/usr/bin/env bash
# A hash warpgroup beside the GEMM's MMA warpgroups on one leased H200 (cwd = the source tree's root). Outside the lease:
# the builds for this GPU (nvcc -Xptxas -v: each kernel's registers and spills) and their SASS. Under one lease of at most
# LEASE_MIN minutes, gates before timing:
#   lowreg   `lowreg check 2` and lowreg_check.py (every kernel's digests against hashlib), then, if it passed, ROUNDS rounds
#            of `lowreg rates LAUNCHES TILES` (SHA-512's rate against the register cap, 1, 2 and 4 warpgroups an SM)
#   beside   `beside gate SHAPE DIR KERNEL` a kernel and beside_check.py (y against the Definition, the handoff's row digests
#            and the hash arm's against hashlib), then, for the kernels that passed, ROUNDS rounds of `beside time SHAPE
#            LAUNCHES KERNEL` (its arms) and Nsight Compute (duty.metrics, the chip's tensor pipes and its warp-stall
#            counters, --clock-control none) on NCU_RUNS (KERNEL:ARM ...)
# Every GPU command is under `timeout STEP_S` (180 s), so a kernel that hangs costs its step and not the lease.
# nvidia-smi samples every 200 ms beside them. BUILD_ONLY=1 stops after the builds (a CPU node with CUDA 13).
#
#   research run --on vy-dc-h200-3 --project verity --campaign hash --source . --cwd source --timeout 60m \
#     --env GPU_LEASE_WHO=kernels-kcommit -- bash benchmarks/torch_gpu/sha512_rows/beside.sh
# STEPS ("lowreg beside"), ROUNDS (3), LAUNCHES (10), TILES (128), SHAPE ("16384 4096 4096": M N K), KERNELS (beside's
# kernels to gate and time: default every one `beside list` names), NCU_RUNS, STEP_S, LEASE_MIN (20), CUDA (the toolkit:
# default the first of /usr/local/cuda-13.0 and /usr/local/cuda).
set -uo pipefail
export O=${OUT:?}
for c in "${CUDA:-}" /usr/local/cuda-13.0 /usr/local/cuda; do [ -n "$c" ] && [ -x "$c/bin/nvcc" ] && { CUDA=$c; break; }; done
export CUDA PATH=$CUDA/bin:$PATH LD_LIBRARY_PATH=$CUDA/lib64:${LD_LIBRARY_PATH:-}
export X=benchmarks/torch_gpu/sha512_rows PY=python3 PYTHONPATH=.
W=$(mktemp -d /dev/shm/beside41.XXXXXX)
export W
trap 'rm -rf "$W"' EXIT
git rev-parse HEAD > "$O/commit.txt" 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}" > "$O/commit.txt"
nvcc --version > "$O/nvcc.txt" 2>&1
export STEPS=${STEPS:-lowreg beside} ROUNDS=${ROUNDS:-3} LAUNCHES=${LAUNCHES:-10} TILES=${TILES:-128}
export SHAPE=${SHAPE:-16384 4096 4096} STEP_S=${STEP_S:-180}
export NCU_RUNS=${NCU_RUNS:-wgl_n256_s4:mma wgl_n128_s3:mma h72_r:mma h72_r:hash h72_r:free h72_r:handoff h64_rs:free \
h128_r:free h128_r_s4:free h128_r_s4:handoff m1h2_r:mma m1h2_r:free m1h3_rs:free m1h4_rs:free m2h2_r_s4:free} KERNELS=${KERNELS:-}
SM=90a
for s in $STEPS; do
  case $s in
    lowreg) nvcc -O3 -std=c++17 -gencode arch=compute_$SM,code=sm_$SM -Xptxas -v -o "$W/lowreg" $X/lowreg.cu -lnvidia-ml \
      > "$O/build-lowreg.log" 2>&1 ;;
    beside) nvcc -O3 -std=c++17 -gencode arch=compute_$SM,code=sm_$SM -fmad=false -Xptxas -v -o "$W/beside" $X/beside.cu \
      -lnvidia-ml > "$O/build-beside.log" 2>&1 ;;
  esac
  echo "build $s rc $?" >> "$O/rc.txt"
  [ -x "$W/$s" ] && cuobjdump -sass "$W/$s" | gzip -c > "$O/sass-$s.txt.gz"
done
cat "$O/rc.txt"
# ptxas's lines a kernel: the demangled entry, its registers, spill stores and loads
for s in $STEPS; do
  [ -s "$O/build-$s.log" ] && $PY - "$O/build-$s.log" > "$O/ptxas-$s.txt" <<'PY'
import re, sys
name = None
for line in open(sys.argv[1]):
    m = re.search(r"Compiling entry function '(\S+)'", line)
    if m:
        name = m.group(1)
    m = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", line)
    if m and name:
        spill = m.groups()
    m = re.search(r"Used (\d+) registers", line)
    if m and name:
        print(f"{name}\tregs {m.group(1)}\tspill st/ld {spill[0]}/{spill[1]}")
        name = None
PY
done
[ "${BUILD_ONLY:-0}" = 1 ] && { head -80 "$O"/ptxas-*.txt; exit 0; }

# shellcheck disable=SC2016
gpu-lease 1 --wait --max-min "${LEASE_MIN:-20}" -- bash -c '
  set -u
  U=$GPU_LEASE_UUID
  nvidia-smi --query-gpu=index,uuid,name,driver_version,compute_cap,clocks.sm,clocks.max.sm,power.limit --format=csv -i "$U" \
    > "$O/gpu.txt" 2>&1
  nvidia-smi -i "$U" -q -d CLOCK,PERFORMANCE > "$O/clock-before.txt" 2>&1
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active \
    --format=csv,noheader -lms 200 -i "$U" > "$O/smi.csv" 2>&1 &
  smi=$!
  start=$SECONDS
  T="timeout -k 10 $STEP_S"
  for s in $STEPS; do
    case $s in
      lowreg)
        [ -x "$W/lowreg" ] || continue
        $T "$W/lowreg" check 2 > "$O/lowreg-check.jsonl" 2> "$O/lowreg-check.err"
        echo "lowreg check rc $? ($((SECONDS - start)) s into the lease)" >> "$O/rc.txt"
        $PY $X/lowreg_check.py "$O/lowreg-check.jsonl" > "$O/lowreg-verdict.txt" 2>&1
        c=$?
        echo "lowreg verdict rc $c" >> "$O/rc.txt"
        [ $c = 0 ] || continue
        for r in $(seq "$ROUNDS"); do
          $T "$W/lowreg" rates "$LAUNCHES" "$TILES" >> "$O/lowreg-rates.jsonl" 2>> "$O/lowreg-rates.err"
          echo "lowreg rates round $r rc $? ($((SECONDS - start)) s into the lease)" >> "$O/rc.txt"
        done ;;
      beside)
        [ -x "$W/beside" ] || continue
        mkdir -p "$W/gate"
        # a process a kernel, so a kernel that hangs or faults costs only its own gate
        for k in ${KERNELS:-$("$W/beside" list)}; do
          # shellcheck disable=SC2086
          $T "$W/beside" gate $SHAPE "$W/gate" "$k" >> "$O/beside-gate.jsonl" 2>> "$O/beside-gate.err"
          echo "beside gate $k rc $? ($((SECONDS - start)) s into the lease)" >> "$O/rc.txt"
        done
        $PY $X/beside_check.py "$O/beside-gate.jsonl" "$W/gate" "$O/beside-passed.txt" > "$O/beside-verdict.txt" 2>&1
        echo "beside verdict rc $?: $(wc -l < "$O/beside-passed.txt") kernel(s) passed" >> "$O/rc.txt"
        rm -rf "$W/gate"
        # timing and profiles only for the kernels whose own gate passed
        passed=$(cat "$O/beside-passed.txt")
        [ -n "$passed" ] || continue
        for r in $(seq "$ROUNDS"); do
          for k in $passed; do
            # shellcheck disable=SC2086
            $T "$W/beside" time $SHAPE "$LAUNCHES" "$k" >> "$O/beside-time.jsonl" 2>> "$O/beside-time.err"
            echo "beside time round $r $k rc $? ($((SECONDS - start)) s into the lease)" >> "$O/rc.txt"
          done
        done
        NCU=$(command -v ncu || ls "$CUDA"/bin/ncu /usr/local/cuda/bin/ncu /usr/local/cuda-*/bin/ncu \
          /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1)
        [ -n "$NCU" ] || { echo "beside ncu rc no-ncu" >> "$O/rc.txt"; continue; }
        "$NCU" --query-metrics --chip "${NCU_CHIP:-gh100}" > "$O/ncu-metrics.txt" 2>&1
        tens=$(awk "{print \$1}" "$O/ncu-metrics.txt" |
          grep -E "^sm__(pipe_tensor[a-z0-9_]*_cycles_active|inst_executed_pipe_tensor[a-z0-9_]*)$" |
          sed "s/$/.avg.pct_of_peak_sustained_elapsed/" | paste -sd, -)
        stall=$( (awk "{print \$1}" "$O/ncu-metrics.txt" | grep -E "^smsp__warps_issue_stalled_[a-z_]+$"
          echo smsp__warps_active) | sed "s/$/.avg.per_cycle_active/" | paste -sd, -)
        base=$(tr -d " \n" < benchmarks/pouw/accel_v31/duty.metrics)
        for kr in $NCU_RUNS; do
          grep -qx "${kr%%:*}" <<< "$passed" || continue
          f="$O/ncu-${kr%%:*}-${kr##*:}"
          for extra in "${tens:+,$tens},$stall" "${tens:+,$tens}" ""; do
            # shellcheck disable=SC2086
            $T "$NCU" --clock-control none --launch-skip 1 --launch-count 1 --metrics "$base$extra" -f -o "$f" \
              "$W/beside" one $SHAPE "${kr%%:*}" "${kr##*:}" 1 > "$f.log" 2>&1
            rc=$?
            [ $rc = 0 ] && break
          done
          echo "beside ncu $kr rc $rc ($((SECONDS - start)) s into the lease)" >> "$O/rc.txt"
          "$NCU" -i "$f.ncu-rep" --csv --page raw --print-units base > "$f.csv" 2>> "$f.log"
        done ;;
    esac
  done
  nvidia-smi -i "$U" -q -d CLOCK > "$O/clock-after.txt" 2>&1
  kill $smi'
echo "lease rc $?" >> "$O/rc.txt"
cat "$O/rc.txt"
tail -1 "$O/lowreg-verdict.txt" 2>/dev/null
tail -1 "$O/beside-verdict.txt" 2>/dev/null
$PY $X/beside_summary.py "$O" | tee "$O/verdict.txt"
