#!/usr/bin/env bash
# GEMMs' board energy a MAC on the H200, at a locked SM clock and at the 700 W cap (cwd: the root of a source tree whose
# accel_v31 fill-bench has the V31_GEN and V31_POWER_* knobs), under `research run`:
#
#   research run --on vy-dc-h200-3 --project verity --campaign CAMPAIGN --source . --cwd source --timeout 60m \
#     --env GPU_LEASE_WHO=LANE -- bash benchmarks/pouw/gemm_energy/energy.sh
#
# Outside the lease: v31.cu's H200 build (and one a VARIANTS entry whose knobs need their own), fill-params for SHAPES,
# and torch for torch_energy.py. Under one gpu-lease (LEASE), for each of CLOCKS (MHz, locked by `sudo -n nvidia-smi
# -lgc C,C`; cap: the driver's clocks under the power limit, `-rgc`):
#   - `v31 fill-bench` on SHAPES for each of VARIANTS (run.sh's knob lists) on uniform int8 operands (V31_GEN=uniform),
#     POWER_REPS rounds of POWER_MS windows over POWER_ARMS (default lt, gemm_d and wgl: cuBLASLt autotuned, the served
#     GEMM of D alone and wgmma.linear), each after POWER_LEAD_MS of the same calls;
#   - torch_energy.py's windows on SHAPES: torch._int_mm, and s8_linear's wgmma.scaled_mm at each of WGMMA_CFGS when
#     WGMMA_CUBIN names its cubin.
# nvidia-smi samples the board throughout (smi.csv), and the clocks are reset when the lease's command exits. After the
# lease: fill-sample on each served call's sampled tiles, and summarize.py's table (table.md, table.json).
#
# SHAPES (default 8B gate-up at M 16384), VARIANTS (default #5613's TMA-store GEMM, hash warps idle), CLOCKS (default
# "1830 cap"), LP (512), FILL_L (4096; L = min(FILL_L, K) through fill-params' --L), REPS (3, fill-bench's timed reps),
# POWER_MS (4000), POWER_LEAD_MS (1000), POWER_REPS (3), POWER_ARMS, IDLE_S (8), WGMMA_CUBIN, WGMMA_CFGS, FLOORS (a
# floors JSON for summarize.py), GENCODE and NVFLAGS (the H200's), LEASE (default gpu-lease 1 --wait --max-min 25 --;
# empty when already under one), PY (v31.py's), TORCH_PY (default $VERITY_BENCH_VENV/bin/python, else uv with torch).
set -uo pipefail
export O=${OUT:?}
export PATH=/usr/local/cuda-13.0/bin:$PATH
D=benchmarks/pouw/accel_v31
E=benchmarks/pouw/gemm_energy
export PYTHONPATH=.
PY=${PY:-uv run -q --no-project --with numpy python}
TORCH_PY=${TORCH_PY:-${VERITY_BENCH_VENV:+$VERITY_BENCH_VENV/bin/python}}
TORCH_PY=${TORCH_PY:-uv run -q --no-project --with torch==2.14.0 --with numpy python}
GENCODE=${GENCODE:-arch=compute_90a,code=sm_90a}
NVFLAGS=${NVFLAGS:--DV31_WG}
SHAPES=${SHAPES:-"8B-up:16384:28672:4096"}
VARIANTS=${VARIANTS:-"served=1,gemm=1,hchunk=8,row=16384,cprio=hi,sched=2,ccta=4,pf=1,lminb=4,sahead=1,rp=5,r5h=96,tst=1,hall=-1"}
export CLOCKS=${CLOCKS:-"1830 cap"} LP=${LP:-512} FILL_L=${FILL_L:-4096} REPS=${REPS:-3}
export POWER_MS=${POWER_MS:-4000} POWER_LEAD_MS=${POWER_LEAD_MS:-1000} POWER_REPS=${POWER_REPS:-3}
export POWER_ARMS=${POWER_ARMS:-lt,gemm_d,wgl} IDLE_S=${IDLE_S:-8} WGMMA_CUBIN=${WGMMA_CUBIN:-} WGMMA_CFGS=${WGMMA_CFGS:-}
export SHAPES VARIANTS TORCH_PY D E
W=$(mktemp -d /dev/shm/gemm-energy.XXXXXX)
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
  trap "sudo -n nvidia-smi ${G[*]} -rgc >> \"$O/clocks.txt\" 2>&1" EXIT
  nvidia-smi "${G[@]}" --query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.max.sm,power.limit,memory.used \
    --format=csv > "$O/gpu.txt" 2>&1
  nvidia-smi "${G[@]}" --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active \
    --format=csv -lms 200 > "$O/smi.csv" 2>&1 &
  smi=$!
  for c in $CLOCKS; do
    mkdir -p "$O/$c" "$W/fb/$c"
    if [ "$c" = cap ]; then sudo -n nvidia-smi "${G[@]}" -rgc; else sudo -n nvidia-smi "${G[@]}" -lgc "$c,$c"; fi \
      >> "$O/clocks.txt" 2>&1
    echo "clock $c rc $?" >> "$O/rc.txt"
    nvidia-smi "${G[@]}" -q -d CLOCK > "$O/$c/clock.txt" 2>&1
    for v in $VARIANTS; do
      tag=$(tr "," "-" <<< "$v" | tr -d "=")
      # shellcheck disable=SC2046,SC2086
      env $(knob_env "$v") V31_GEN=uniform V31_POWER_MS="$POWER_MS" V31_POWER_LEAD_MS="$POWER_LEAD_MS" \
        V31_POWER_REPS="$POWER_REPS" V31_POWER_ARMS="$POWER_ARMS" "$W/v31$(vbin "$v")" fill-bench "$REPS" "$W/fb/$c/$tag" \
        "$W/fparams" "$LP" "$FILL_L" "$(knob "$v" row 1024)" $SHAPES > "$O/$c/fb-$tag.jsonl" 2> "$O/$c/fb-$tag.err"
      echo "fill-bench $c $tag rc $?" >> "$O/rc.txt"
    done
    a=(); for s in $SHAPES; do a+=(--shape "$s"); done
    [ -n "$WGMMA_CUBIN" ] && a+=(--wgmma-cubin "$WGMMA_CUBIN") && for g in $WGMMA_CFGS; do a+=(--cfg "$g"); done
    # shellcheck disable=SC2086
    $TORCH_PY $E/torch_energy.py "${a[@]}" --window-ms "$POWER_MS" --lead-ms "$POWER_LEAD_MS" --rounds "$POWER_REPS" \
      --idle-s "$IDLE_S" > "$O/$c/torch.jsonl" 2> "$O/$c/torch.err"
    echo "torch $c rc $?" >> "$O/rc.txt"
  done
  kill $smi'
echo "lease rc $?" >> "$O/rc.txt"

for s in "$W"/fb/*/*/samples/*/; do
  [ -d "$s" ] || continue
  name=$(echo "${s#"$W"/fb/}" | tr "/" "-" | sed "s/-samples-/-/; s/-$//")
  $PY $D/v31.py fill-sample "$s" > "$O/fill-sample-$name.out" 2> "$O/fill-sample-$name.err"
  echo "fill-sample $name rc $?" >> "$O/rc.txt"
done
f=(); [ -n "${FLOORS:-}" ] && f=(--floors "$FLOORS")
python3 $E/summarize.py "$O" "${f[@]}" --json "$O/table.json" > "$O/table.md" 2> "$O/table.err"
echo "summarize rc $?" >> "$O/rc.txt"
cat "$O/rc.txt" "$O/table.md"
! grep -qv ' rc 0$' "$O/rc.txt"
