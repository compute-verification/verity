#!/usr/bin/env bash
# int8 GEMMs' MACs per SM clock on the H200, against 4,096, in sustained windows on uniform operands (cwd: the root of a
# source tree whose accel_v31 fill-bench has the V31_GEN and V31_WINDOW_* knobs), under `research run`:
#
#   research run --on MACHINE --project verity --campaign CAMPAIGN --source . --cwd source --timeout 60m \
#     --env GPU_LEASE_WHO=LANE -- bash benchmarks/pouw/gemm_rate/rate.sh
#
# Outside the lease: v31.cu's H200 build (and one a VARIANTS entry whose knobs need their own), stamp.cu's cubin,
# s8_linear's cubin (build.sh's) when WGMMA_CFGS names configs, fill-params for SHAPES, and torch for torch_rate.py. Under
# one gpu-lease (LEASE), at the card's own clocks:
#   - `v31 fill-bench` on SHAPES for each of VARIANTS (run.sh's knob lists) on uniform int8 operands (V31_GEN=uniform),
#     WINDOW_REPS rounds of WINDOW_MS windows over WINDOW_ARMS (default lt, gemm_d and wgl: cuBLASLt autotuned, the served
#     GEMM of D alone and the wgmma.linear port), each after WINDOW_LEAD_MS of the same calls, each between two stamps of
#     every SM;
#   - torch_rate.py's windows on SHAPES: torch._int_mm, and s8_linear's wgmma.scaled_mm at each of WGMMA_CFGS.
# nvidia-smi samples the SM clock throughout (smi.csv). After the lease: fill-sample on each served call's sampled tiles,
# and summarize.py's table (table.md, table.json).
#
# SHAPES (default 8B gate-up at M 16384), VARIANTS (default the served TMA-store GEMM, tst=1, hash warps idle), LP (512),
# FILL_L (4096; L = min(FILL_L, K) through fill-params' --L), REPS (3, fill-bench's timed reps), WINDOW_MS (2000),
# WINDOW_LEAD_MS (500), WINDOW_REPS (3), WINDOW_ARMS, WGMMA_CFGS (none), GENCODE and NVFLAGS (the H200's), LEASE (default
# gpu-lease 1 --wait --max-min 25 --; empty when already under one), PY (v31.py's), TORCH_PY (default
# $VERITY_BENCH_VENV/bin/python, else uv with torch).
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/pouw/accel_v31
E=benchmarks/pouw/gemm_rate
export PYTHONPATH=.:tools/config
PY=${PY:-uv run -q --no-project --with numpy python}
TORCH_PY=${TORCH_PY:-${VERITY_BENCH_VENV:+$VERITY_BENCH_VENV/bin/python}}
TORCH_PY=${TORCH_PY:-uv run -q --no-project --with torch==2.14.0 --with numpy python}
GENCODE=${GENCODE:-arch=compute_90a,code=sm_90a}
NVFLAGS=${NVFLAGS:--DV31_WG}
SHAPES=${SHAPES:-"8B-up:16384:28672:4096"}
VARIANTS=${VARIANTS:-"served=1,gemm=1,hchunk=8,row=16384,cprio=hi,sched=2,ccta=4,pf=1,lminb=4,sahead=1,rp=5,r5h=96,tst=1,hall=-1"}
export LP=${LP:-512} FILL_L=${FILL_L:-4096} REPS=${REPS:-3} WINDOW_MS=${WINDOW_MS:-2000}
export WINDOW_LEAD_MS=${WINDOW_LEAD_MS:-500} WINDOW_REPS=${WINDOW_REPS:-3} WINDOW_ARMS=${WINDOW_ARMS:-lt,gemm_d,wgl}
export WGMMA_CFGS=${WGMMA_CFGS:-} SHAPES VARIANTS TORCH_PY D E
W=$(mktemp -d /dev/shm/gemm-rate.XXXXXX)
export W
trap 'rm -rf "$W"' EXIT
git rev-parse HEAD > "$O/commit.txt" 2>&1
: > "$O/rc.txt"
# run.sh's knob, knob_env, vbin, vflags and vsha, so a variant here means what it means there
# shellcheck disable=SC1090
source <(sed -n '/^knob() {/,/^export -f knob /p' $D/run.sh)
declare -F knob_env vbin vflags vsha > /dev/null || { echo "no knob functions in $D/run.sh" | tee -a "$O/rc.txt"; exit 1; }

build() {   # BIN LOG [nvcc flags] (BSHA: sha512.cuh's flags in place of run.sh's default)
  # shellcheck disable=SC2086
  nvcc -O3 -std=c++17 -gencode "$GENCODE" -Xptxas -v ${BSHA:--DSHA_ADD=0 -DSHA_ROT=0 -DSHA_ROLL=1} $NVFLAGS "${@:3}" \
    -o "$1" $D/v31.cu -lcuda -lcublasLt > "$2" 2>&1
}
build "$W/v31" "$O/build.log"
echo "build rc $?" >> "$O/rc.txt"
for v in $VARIANTS; do
  b=$(vbin "$v")
  { [ -z "$b" ] || [ -e "$W/v31$b" ]; } && continue
  # shellcheck disable=SC2046
  BSHA=$(vsha "$v") build "$W/v31$b" "$O/build$b.log" $(vflags "$v")
  echo "build ${b#-} rc $?" >> "$O/rc.txt"
done
nvcc -O3 -gencode "$GENCODE" -cubin -o "$W/stamp.cubin" $E/stamp.cu > "$O/build-stamp.log" 2>&1
echo "build stamp rc $?" >> "$O/rc.txt"
if [ -n "$WGMMA_CFGS" ]; then
  PY=/nonexistent bash benchmarks/torch_gpu/s8_linear/build.sh "$W/s8" > "$O/build-s8.log" 2>&1
  echo "build s8_wgmma rc $?" >> "$O/rc.txt"
fi
for v in $VARIANTS; do
  # shellcheck disable=SC2086
  $PY $D/v31.py fill-params "$W/fparams" $SHAPES --lp "$LP" --L "$FILL_L" --row "$(knob "$v" row 1024)" --served \
    2>> "$O/prep.err"
  echo "fill-params $(knob "$v" row 1024) rc $?" >> "$O/rc.txt"
done
$TORCH_PY -c 'import torch; print(torch.__version__, torch.version.cuda)' > "$O/torch.txt" 2>&1
echo "torch rc $?" >> "$O/rc.txt"
grep -qv ' rc 0$' "$O/rc.txt" && { cat "$O/rc.txt"; tail -20 "$O/build.log" "$O/prep.err"; exit 1; }

LEASE=${LEASE-gpu-lease 1 --wait --max-min 25 --}
# shellcheck disable=SC2016
$LEASE bash -c '
  G=(); [ -n "${GPU_LEASE_UUID:-}" ] && G=(-i "$GPU_LEASE_UUID")
  nvidia-smi "${G[@]}" --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,memory.used \
    --format=csv > "$O/gpu.txt" 2>&1
  nvidia-smi "${G[@]}" --query-gpu=timestamp,clocks.sm,temperature.gpu,clocks_event_reasons.active \
    --format=csv -lms 500 > "$O/smi.csv" 2>&1 &
  smi=$!
  for v in $VARIANTS; do
    tag=$(tr "," "-" <<< "$v" | tr -d "=")
    # shellcheck disable=SC2046,SC2086
    env $(knob_env "$v") V31_GEN=uniform V31_POWER_MS=0 V31_WINDOW_MS="$WINDOW_MS" V31_WINDOW_LEAD_MS="$WINDOW_LEAD_MS" \
      V31_WINDOW_REPS="$WINDOW_REPS" V31_WINDOW_ARMS="$WINDOW_ARMS" "$W/v31$(vbin "$v")" fill-bench "$REPS" "$W/fb/$tag" \
      "$W/fparams" "$LP" "$FILL_L" "$(knob "$v" row 1024)" $SHAPES > "$O/fb-$tag.jsonl" 2> "$O/fb-$tag.err"
    echo "fill-bench $tag rc $?" >> "$O/rc.txt"
  done
  a=(); for s in $SHAPES; do a+=(--shape "$s"); done
  [ -n "$WGMMA_CFGS" ] && a+=(--wgmma-cubin "$W/s8/s8_wgmma.cubin") && for g in $WGMMA_CFGS; do a+=(--cfg "$g"); done
  $TORCH_PY $E/torch_rate.py --stamp-cubin "$W/stamp.cubin" "${a[@]}" --window-ms "$WINDOW_MS" \
    --lead-ms "$WINDOW_LEAD_MS" --rounds "$WINDOW_REPS" > "$O/torch.jsonl" 2> "$O/torch.err"
  echo "torch rc $?" >> "$O/rc.txt"
  kill $smi'
echo "lease rc $?" >> "$O/rc.txt"

for s in "$W"/fb/*/samples/*/; do
  [ -d "$s" ] || continue
  name=$(echo "${s#"$W"/fb/}" | tr "/" "-" | sed "s/-samples-/-/; s/-$//")
  $PY $D/v31.py fill-sample "$s" > "$O/fill-sample-$name.out" 2> "$O/fill-sample-$name.err"
  echo "fill-sample $name rc $?" >> "$O/rc.txt"
done
python3 $E/summarize.py "$O" --json "$O/table.json" > "$O/table.md" 2> "$O/table.err"
echo "summarize rc $?" >> "$O/rc.txt"
cat "$O/rc.txt" "$O/table.md"
! grep -qv ' rc 0$' "$O/rc.txt"
