#!/usr/bin/env bash
# The tensor-duty sweep (duty.cu) on one H200. Builds outside the lease, in a scratch directory under the run's own; then,
# under one gpu-lease (the job's own when it already holds one, GPU_LEASE_UUID), sets the leased GPU's power limit to the
# board's maximum (-pl) and, for each of CLOCKS, locks its SM clock there (nvidia-smi -lgc; `default` resets it, -rgc),
# warms the card and runs every arm, class, fill, burst and duty. A clock whose lock fails is skipped; the default clock
# still runs. On exit it resets the clock and puts the power limit back to the board's default. nvidia-smi samples the
# GPU every SMI_MS ms (200) while the lease is held. Last, summarize.py judges the H200 card's clock.tensor by the
# points, and the run exits nonzero unless it passes.
#
#   research run --on NODE --project verity --source . --cwd source --env GPU_LEASE_WHO=LANE \
#     -- bash benchmarks/tensor_duty/run.sh [--arms fp32-beside] [--no-lock]
#   PYTHONPATH=. python3 benchmarks/tensor_duty/summarize.py ~/.research/runs/RUN/out
#
# --arms fp32-beside: in place of the duty sweep, FP32 adds beside the tensor pipe (duty.cu's arms A to D) at each clock,
# for each of BESIDE_CLASSES ("zero const uniform"): A, B, C at the add:MAC ratios 1/X for X in RATIOS ("128 32 8"), and D
# in phases of PHASES_MS ("1 10 100"), each point SECS; the warm-up is arm A on uniform operands; nvidia-smi samples every
# 10 ms; summarize.py --arms fp32-beside writes summary.json, a verdict per reading of an FP32 add's price, and
# exits nonzero unless every reading is decided (BASE_MHZ and BASE_SOURCE: the cautious reading's base clock).
# --no-lock (LOCK=0): no sudo; the clock and power limit stay as found and CLOCKS is `default`.
#
# Env: CLOCKS (required unless --no-lock, which sets it to `default`: the SM clocks to lock in MHz, the H200 card's
# clock.sm, then `default`); ARMS ("wg_s8 wg_e4m3 sync_s8"); each arm's classes and bursts (SM-cycles of tensor work
# a period): WG_S8_CLASSES ("zero const uniform s2 u2 u4"), WG_S8_BURSTS ("1024 65536 2097152"), WG_E4M3_CLASSES
# ("zero const uniform"), WG_E4M3_BURSTS ("4096 65536"), SYNC_S8_CLASSES ("zero uniform"), SYNC_S8_BURSTS ("65536");
# FILLS ("idle alu"); DUTIES (50 to 90 by 5, then 91 to 100, in percent); SECS (1.2, each point's kernel); WARM_S (20:
# continuous wg_s8 on uniform operands, at each clock first); LEASE_MIN (75); BUILD_ONLY (1: build and list the
# binary's tensor and FADD instructions, no lease).
# Writes $OUT/points.jsonl (beside.jsonl with --arms fp32-beside) and warm.jsonl (duty's lines, each with its clock),
# smi.csv, found.json, gpu-before.txt, gpu-after.txt, lock.txt, clock-CLOCK-before.txt and -locked.txt (nvidia-smi -q -d
# CLOCK,PERFORMANCE before and after each clock's lock; -after.txt after its fp32-beside arms), restored.txt, rc.txt,
# build.log, sass_ops.txt, and summarize.py's summary.md and verdict.json (summary.json with --arms fp32-beside).
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd)
O=${OUT:?run under research run}
while [ $# -gt 0 ]; do
  case $1 in
    --arms) ARMS=$2; shift 2 ;;
    --no-lock) LOCK=0; shift ;;
    *) break ;;
  esac
done
export PATH=/usr/local/cuda/bin:$PATH CUDA_MODULE_LOADING=EAGER
export ARMS=${ARMS:-wg_s8 wg_e4m3 sync_s8} FILLS=${FILLS:-idle alu} SECS=${SECS:-1.2} WARM_S=${WARM_S:-20}
export WG_S8_CLASSES=${WG_S8_CLASSES:-zero const uniform s2 u2 u4} WG_S8_BURSTS=${WG_S8_BURSTS:-1024 65536 2097152}
export WG_E4M3_CLASSES=${WG_E4M3_CLASSES:-zero const uniform} WG_E4M3_BURSTS=${WG_E4M3_BURSTS:-4096 65536}
export SYNC_S8_CLASSES=${SYNC_S8_CLASSES:-zero uniform} SYNC_S8_BURSTS=${SYNC_S8_BURSTS:-65536}
export DUTIES=${DUTIES:-50 55 60 65 70 75 80 85 90 91 92 93 94 95 96 97 98 99 100} O
export BESIDE_CLASSES=${BESIDE_CLASSES:-zero const uniform} RATIOS=${RATIOS:-128 32 8} PHASES_MS=${PHASES_MS:-1 10 100}
export LOCK=${LOCK:-1}
[ "$LOCK" = 1 ] || CLOCKS=default
export CLOCKS=${CLOCKS:?set the SM clocks to lock in MHz then default, clock.sm of the H200 card, or pass --no-lock}
[ "$ARMS" = fp32-beside ] && SMI_MS=${SMI_MS:-10}
export SMI_MS=${SMI_MS:-200}

if [ "${1:-}" = --leased ]; then
  BIN=$2
  UUID=${GPU_LEASE_UUID:?under gpu-lease}
  q() { nvidia-smi -i "$UUID" --query-gpu="$1" --format=csv,noheader,nounits | tr -d ' '; }
  DEF_PL=$(q power.default_limit)
  MAX_PL=$(q power.max_limit)
  printf '{"uuid": "%s", "name": "%s", "max_sm_mhz": "%s", "found_pl_w": "%s", "default_pl_w": "%s", "max_pl_w": "%s", "clocks": "%s", "lock": "%s", "arms": "%s", "tz": "%s"}\n' \
    "$UUID" "$(nvidia-smi -i "$UUID" --query-gpu=name --format=csv,noheader)" "$(q clocks.max.sm)" "$(q power.limit)" \
    "$DEF_PL" "$MAX_PL" "$CLOCKS" "$LOCK" "$ARMS" "$(date +%z)" > "$O/found.json"
  nvidia-smi -i "$UUID" -q -d CLOCK,POWER,PERFORMANCE,TEMPERATURE > "$O/gpu-before.txt" 2>&1
  nvidia-smi -i "$UUID" --query-gpu=timestamp,clocks.sm,clocks.applications.graphics,power.draw,power.limit,temperature.gpu,clocks_event_reasons.active \
    --format=csv,noheader,nounits -lms "$SMI_MS" > "$O/smi.csv" 2>&1 &
  SMI=$!
  restore() {
    if [ "$LOCK" = 1 ]; then
      { sudo -n nvidia-smi -i "$UUID" -rgc; echo "rgc rc $?"; sudo -n nvidia-smi -i "$UUID" -pl "$DEF_PL"; echo "pl rc $?"; } \
        > "$O/restored.txt" 2>&1
    fi
    kill "$SMI" 2>/dev/null
    nvidia-smi -i "$UUID" -q -d CLOCK,POWER,PERFORMANCE,TEMPERATURE > "$O/gpu-after.txt" 2>&1
  }
  trap restore EXIT
  trap 'exit 143' TERM INT
  if [ "$LOCK" = 1 ]; then
    { sudo -n nvidia-smi -i "$UUID" -pl "$MAX_PL"; echo "pl rc $?"; } > "$O/lock.txt" 2>&1
  else
    echo "no lock: clock and power limit as found" > "$O/lock.txt"
  fi
  for clock in $CLOCKS; do
    nvidia-smi -i "$UUID" -q -d CLOCK,PERFORMANCE > "$O/clock-$clock-before.txt" 2>&1
    if [ "$LOCK" != 1 ]; then true
    elif [ "$clock" = default ]; then sudo -n nvidia-smi -i "$UUID" -rgc
    else sudo -n nvidia-smi -i "$UUID" -lgc "$clock,$clock"; fi >> "$O/lock.txt" 2>&1
    rc=$?
    echo "clock $clock rc $rc" | tee -a "$O/lock.txt" >> "$O/rc.txt"
    nvidia-smi -i "$UUID" --query-gpu=clocks.sm,clocks.applications.graphics,power.limit,enforced.power.limit --format=csv \
      >> "$O/lock.txt" 2>&1
    nvidia-smi -i "$UUID" -q -d CLOCK,PERFORMANCE > "$O/clock-$clock-locked.txt" 2>&1
    if [ "$clock" != default ] && [ "$rc" != 0 ]; then
      echo "clock $clock skipped: not locked" >> "$O/rc.txt"
      continue
    fi
    tag="{\"clock\": \"$clock\", "
    if [ "$ARMS" = fp32-beside ]; then
      "$BIN" A uniform "$WARM_S" 0 2>> "$O/duty.err" | sed "s/^{/$tag/" >> "$O/warm.jsonl"
      for c in $BESIDE_CLASSES; do
        for arm in A B C D; do
          case $arm in C) xs=$RATIOS ;; D) xs=$PHASES_MS ;; *) xs=0 ;; esac
          "$BIN" "$arm" "$c" "$SECS" $xs 2>> "$O/duty.err" | sed "s/^{/$tag/" >> "$O/beside.jsonl"
          echo "$clock $arm $c rc ${PIPESTATUS[0]}" >> "$O/rc.txt"
        done
      done
      nvidia-smi -i "$UUID" -q -d CLOCK,PERFORMANCE > "$O/clock-$clock-after.txt" 2>&1
      continue
    fi
    "$BIN" wg_s8 uniform idle 65536 "$WARM_S" 100 2>> "$O/duty.err" | sed "s/^{/$tag/" >> "$O/warm.jsonl"
    for arm in $ARMS; do
      case $arm in
        wg_s8) classes=$WG_S8_CLASSES bursts=$WG_S8_BURSTS ;;
        wg_e4m3) classes=$WG_E4M3_CLASSES bursts=$WG_E4M3_BURSTS ;;
        sync_s8) classes=$SYNC_S8_CLASSES bursts=$SYNC_S8_BURSTS ;;
        *) echo "unknown arm $arm" >> "$O/rc.txt"; continue ;;
      esac
      for c in $classes; do
        for f in $FILLS; do
          for b in $bursts; do
            "$BIN" "$arm" "$c" "$f" "$b" "$SECS" $DUTIES 2>> "$O/duty.err" | sed "s/^{/$tag/" >> "$O/points.jsonl"
            echo "$clock $arm $c $f $b rc ${PIPESTATUS[0]}" >> "$O/rc.txt"
          done
        done
      done
    done
  done
  exit 0
fi

S=$(mktemp -d -p "$(dirname "$O")" scratch.XXXXXX)
trap 'rm -rf "$S"' EXIT
export TMPDIR=$S PYTHONDONTWRITEBYTECODE=1
nvcc --version > "$O/nvcc.txt" 2>&1
nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -Xptxas -v -I"$D" -I"$D/../int8_class_rate" -o "$S/duty" \
  "$D/duty.cu" > "$O/build.log" 2>&1
echo "build rc $?" | tee -a "$O/rc.txt"
[ -x "$S/duty" ] || { tail -40 "$O/build.log"; exit 1; }
cuobjdump -sass "$S/duty" | grep -oE "(IGMMA|QGMMA|HGMMA|IMMA|FADD)[.A-Za-z0-9_]*" | sort | uniq -c | tee "$O/sass_ops.txt"
[ "${BUILD_ONLY:-0}" = 1 ] && exit 0
if [ -n "${GPU_LEASE_UUID:-}" ]; then
  bash "$D/run.sh" --leased "$S/duty"
else
  gpu-lease 1 --wait --max-min "${LEASE_MIN:-75}" -- bash "$D/run.sh" --leased "$S/duty"
fi
echo "lease rc $?" | tee -a "$O/rc.txt"
if [ "$ARMS" = fp32-beside ]; then
  PYTHONPATH="$D/../.." python3 "$D/summarize.py" "$O" --arms fp32-beside --json "$O/summary.json" \
    ${BASE_MHZ:+--base-mhz "$BASE_MHZ" --base-source "${BASE_SOURCE:?with BASE_MHZ}"} | tee "$O/summary.md"
else
  PYTHONPATH="$D/../.." python3 "$D/summarize.py" "$O" --json "$O/verdict.json" | tee "$O/summary.md"
fi
