#!/usr/bin/env bash
# A hash warpgroup beside the GEMM's MMA warpgroups on one leased H200 (cwd = the source tree's root). Outside the lease:
# the builds for this GPU (nvcc -Xptxas -v: each kernel's registers and spills) and their SASS. Under one lease of at most
# LEASE_MIN minutes, gates before timing:
#   lowreg   `lowreg check 2` and lowreg_check.py (every kernel's digests against hashlib), then, if it passed, ROUNDS rounds
#            of `lowreg rates LAUNCHES TILES` (SHA-512's rate against the register cap, 1, 2 and 4 warpgroups an SM)
# nvidia-smi samples every 200 ms beside them. BUILD_ONLY=1 stops after the builds (a CPU node with CUDA 13).
#
#   research run --on vy-dc-h200-3 --project verity --campaign hash --source . --cwd source --timeout 60m \
#     --env GPU_LEASE_WHO=kernels-kcommit -- bash benchmarks/torch_gpu/sha512_rows/beside.sh
# STEPS ("lowreg"), ROUNDS (3), LAUNCHES (10), TILES (128), LEASE_MIN (20), CUDA (the toolkit: default the first of
# /usr/local/cuda-13.0 and /usr/local/cuda).
set -uo pipefail
export O=${OUT:?}
for c in "${CUDA:-}" /usr/local/cuda-13.0 /usr/local/cuda; do [ -n "$c" ] && [ -x "$c/bin/nvcc" ] && { CUDA=$c; break; }; done
export PATH=$CUDA/bin:$PATH LD_LIBRARY_PATH=$CUDA/lib64:${LD_LIBRARY_PATH:-}
export X=benchmarks/torch_gpu/sha512_rows PY=python3 PYTHONPATH=.
W=$(mktemp -d /dev/shm/beside41.XXXXXX)
export W
trap 'rm -rf "$W"' EXIT
git rev-parse HEAD > "$O/commit.txt" 2>/dev/null || echo "${RESEARCH_SOURCE_SHA:-unknown}" > "$O/commit.txt"
nvcc --version > "$O/nvcc.txt" 2>&1
export STEPS=${STEPS:-lowreg} ROUNDS=${ROUNDS:-3} LAUNCHES=${LAUNCHES:-10} TILES=${TILES:-128}
SM=90a
for s in $STEPS; do
  case $s in
    lowreg) nvcc -O3 -std=c++17 -gencode arch=compute_$SM,code=sm_$SM -Xptxas -v -o "$W/lowreg" $X/lowreg.cu -lnvidia-ml \
      > "$O/build-lowreg.log" 2>&1 ;;
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
  for s in $STEPS; do
    case $s in
      lowreg)
        [ -x "$W/lowreg" ] || continue
        "$W/lowreg" check 2 > "$O/lowreg-check.jsonl" 2> "$O/lowreg-check.err"
        echo "lowreg check rc $? ($((SECONDS - start)) s into the lease)" >> "$O/rc.txt"
        $PY $X/lowreg_check.py "$O/lowreg-check.jsonl" > "$O/lowreg-verdict.txt" 2>&1
        c=$?
        echo "lowreg verdict rc $c" >> "$O/rc.txt"
        [ $c = 0 ] || continue
        for r in $(seq "$ROUNDS"); do
          "$W/lowreg" rates "$LAUNCHES" "$TILES" >> "$O/lowreg-rates.jsonl" 2>> "$O/lowreg-rates.err"
          echo "lowreg rates round $r rc $? ($((SECONDS - start)) s into the lease)" >> "$O/rc.txt"
        done ;;
    esac
  done
  nvidia-smi -i "$U" -q -d CLOCK > "$O/clock-after.txt" 2>&1
  kill $smi'
echo "lease rc $?" >> "$O/rc.txt"
cat "$O/rc.txt"
tail -1 "$O/lowreg-verdict.txt" 2>/dev/null
[ -s "$O/lowreg-rates.jsonl" ] && $PY $X/beside_summary.py "$O" | tee "$O/verdict.txt"
