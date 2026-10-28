#!/usr/bin/env bash
# The honest pipelined Dither GEMM on one B200 (pgemm.cu): its gates at L = 32 and 64, then its rate at L = 32, 64, 128
# and 256 in each arm, against ρ_T5. From a pushed commit, on a 1x B200 node:
#
#   research run --on <node> --project verity --source <tree> --cwd source --custody-ttl 12h \
#     --timeout 40m --env GPU_LEASE_WHO=dc -- bash benchmarks/pouw/dc/pipe.sh
#
# Outside the lease: build pgemm (build.log, ptxas -v), dump its SASS (sass/pgemm.sass) and refuse a kernel that spills or
# lacks UTCIMMA or LDTM.x32 (gate-build.txt). Under one lease, with nvidia-smi sampled every 200 ms (env/smi.csv): the
# honest kernel (MODE 2) on every (NB, LW) at lm 1 and 2 on two shapes, each dump rechecked by dgemm_ref.py
# (gate/<shape>/result.json, ref.json); the rate of every (NB, LW, arm, lm) only if every gate is exact (rate.jsonl);
# pipe.py analyze (pipe.json, pipe.txt).
# Exit: 0 all done, 1 build refused, 4 a gate not exact (no rate), 2 wrong device or no nvcc.
# DC_BUILD_ONLY=1 stops after the build and SASS gate and needs no GPU (any machine with CUDA 13.0).
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "pipe.sh: no nvcc (CUDA 13.0 has sm_100a)" >&2; exit 2; }
export NVCC PATH="$(dirname "$NVCC"):$PATH"
TILES=("2 1" "2 2" "4 1" "4 2")
SHAPES=("128 512 256 7" "256 1024 1024 11")
REPS=${PIPE_REPS:-7}
KSTEPS=${PIPE_KSTEPS:-2048}

if [ "${DC_LEASED:-}" != 1 ]; then
  WORK=$(mktemp -d)
  trap 'rm -rf "$WORK"' EXIT
  mkdir -p "$OUT/sass" "$OUT/env" "$OUT/gate"
  "$NVCC" --version >"$OUT/env/nvcc.txt" 2>&1
  if [ "${DC_BUILD_ONLY:-}" != 1 ]; then
    nvidia-smi -q >"$OUT/env/nvidia-smi.txt" 2>&1
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
    [ "${cap%%.*}" = 10 ] || { echo "pipe.sh: needs compute capability 10.x, the device reports '$cap'" >&2; exit 2; }
  fi
  "$NVCC" -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -Xptxas -v "$HERE/pgemm.cu" -o "$WORK/pgemm" \
    >"$OUT/build.log" 2>&1 || { cat "$OUT/build.log" >&2; exit 1; }
  cuobjdump -sass "$WORK/pgemm" >"$OUT/sass/pgemm.sass" 2>&1
  python3 "$HERE/pipe.py" gate-build "$OUT" >"$OUT/gate-build.txt"
  rc=$?
  grep -c '^PASS' "$OUT/gate-build.txt" | xargs echo "pipe.sh: gate-build PASS"
  [ $rc -eq 0 ] || { cat "$OUT/gate-build.txt" >&2; exit 1; }
  [ "${DC_BUILD_ONLY:-}" = 1 ] && exit 0

  lease=()
  command -v gpu-lease >/dev/null && lease=(gpu-lease 1 --wait --max-min 30 --)
  [ ${#lease[@]} -eq 0 ] && echo "pipe.sh: no gpu-lease on this node; running on its one GPU directly" >&2
  "${lease[@]}" env DC_LEASED=1 WORK="$WORK" bash "${BASH_SOURCE[0]}"
  exit $?
fi

# ---- leased ------------------------------------------------------------------------------------------------------------
nvidia-smi --query-gpu=timestamp,uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active \
  --format=csv -lms 200 >"$OUT/env/smi.csv" 2>&1 &
smi_pid=$!
trap 'kill "$smi_pid" 2>/dev/null' EXIT

exact=1
for mi in 0 1; do
  for t in "${TILES[@]}"; do
    read -r nb lw <<<"$t"
    for lm in 1 2; do
      for s in "${SHAPES[@]}"; do
        read -r m n k seed <<<"$s"
        d="$OUT/gate/${m}x${n}x${k}-L$((32 * lm))-nb${nb}-lw${lw}-mi${mi}"
        mkdir -p "$d"
        "$WORK/pgemm" gate "$d" "$m" "$n" "$k" "$seed" "$lm" "$nb" "$lw" "$mi" | tee "$d/result.json" || exact=0
        python3 "$HERE/dgemm_ref.py" "$d" | tee "$d/ref.json" || exact=0
        rm -f "$d"/*.bin
      done
    done
  done
done
if [ $exact = 1 ]; then
  "$WORK/pgemm" rate "$REPS" "$KSTEPS" | tee "$OUT/rate.jsonl"
  python3 "$HERE/pipe.py" analyze "$OUT" --json "$OUT/pipe.json" | tee "$OUT/pipe.txt"
else
  echo "pipe.sh: a gate is not exact; no rate" >&2
  exit 4
fi
