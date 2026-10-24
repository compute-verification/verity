#!/usr/bin/env bash
# Dither's served prefill row on vy-nebius-2: sbench.sh --prefill --serial --t $T (2048) with exact, exact-fused and $ROW,
# the decode row's units and table builds (HB_KC_FLAGS, HB_KC_H1C_FLAGS) and window flags (DITHER_SO_FUSED: a step's
# units one h1um launch), except h1um's L2 prefetch: L2PF (0) is H1U_L2PF.  At prefill's m = 2048 every A and B line a
# unit reads is L2's already, and the prefetches cost h1um 2.3 % (unitsbench --multi, step_multi 397.6 -> 388.6 ms;
# at decode's m = 32 6.425 -> 6.414 ms).  KCX and KHX append to the two builds' flags, EXTRA to sbench.py's arguments.
#   research run --on vy-nebius-2 --project verity --source . --cwd source --env GPU_LEASE_WHO=LANE --env SBENCH_ON=4,5,6,7 \
#     --env NUMA=1 --env SPLIT=1 --env SBENCH_LEASE_AT=2026-10-24T05:15:30Z --env GPU_LEASE_TIMED=1 \
#     -- bash benchmarks/hash/dither/prefill.sh
# SPLIT=1 runs two passes over the same arms and flags: out/gate, every arm gated and none timed, under an untimed lease
# of GATE_MAX_MIN (15) minutes asked at once, so it ends before a clean window opens; then out/time, every arm timed REPS
# (3) times and none gated, under the lease SBENCH_LEASE_AT and GPU_LEASE_TIMED ask for (the window's).  Without it, one
# pass into out/ (--gate unless GATE=0, REPS 1).  NUMA=n binds every process to node n's CPUs and memory (n2's GPUs 4-7
# are node 1's); out/load.csv samples the node's load every 10 s.
set -uo pipefail
D=${RESEARCH_RUN_DIR:?}
mkdir -p "$D/out"
export HB_KC_FLAGS="-DH1U_L2HINT=128 -DH1U_WARPS=2 -DH1U_EVF=1 -DH1S_EVL=1 -DH1U_EVFU=0 -DH1K_CPA=3 -DH1U_PF=1 -DH1U_NB2=1 -DH1U_L2PF=${L2PF:-0} ${KCX:-}"
export HB_KC_H1C_FLAGS="-DH1C_HB=1 -DH1C_MK=1 -DH1C_MID=1 -DH1C_DOMSEG=1 -DH1C_LVF=2 -DH1C_LMINB=4 -DH1C_APW=1 -DH1C_SEGMID=1 ${KHX:-}"
export DITHER_SO_FUSED=1 DITHER_ZDEAD=1 DITHER_SO_RDEAD=1 DITHER_A_STEP=1 DITHER_SO_COIN_ASYNC=2 DITHER_SO_PRIO=1 DITHER_ZY1_KC=1
export DITHER_SPILL_MAX_GB=${DITHER_SPILL_MAX_GB:-96} SBENCH_MAX_MIN=${SBENCH_MAX_MIN:-20}
export DITHER_CHECK_WORKERS=${DITHER_CHECK_WORKERS:-16}
ROW=${ROW:-dither-a-hbc-zy-aom-pt-so3-wea}
read -r -a extra <<< "${EXTRA:-}"
# exact at 0.14 of the GPU; Dither's window at T = 2048 needs 0.88, so --serial: one arm's engine on the GPU at a time
BASE=(--prefill --serial --t "${T:-2048}" --arms "${ARMS:-exact,exact-fused,$ROW}" --gpu-mem "${GPU_MEM:-0.14}"
      --arm-mem "$ROW=${ROW_MEM:-0.88}" --ops "${OPS:-prefill}" --time-passed)
echo "prefill: HB_KC_FLAGS=$HB_KC_FLAGS"
echo "prefill: HB_KC_H1C_FLAGS=$HB_KC_H1C_FLAGS"
BIND=()
if [ -n "${NUMA:-}" ]; then
  BIND=(numactl --cpunodebind="$NUMA" --membind="$NUMA")
  echo "prefill: NUMA=$NUMA: $("${BIND[@]}" numactl --show | tr '\n' ' ')"
fi
( while :; do
    echo "$(date -u +%H:%M:%S) $(cut -d' ' -f1-3 /proc/loadavg) $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ';')"
    sleep 10
  done ) > "$D/out/load.csv" &
sampler=$!
if [ "${SPLIT:-0}" = 1 ]; then
  echo "prefill: gate pass ${BASE[*]} --reps 0 --gate ${extra[*]}"
  env -u SBENCH_LEASE_AT -u GPU_LEASE_TIMED SBENCH_MAX_MIN="${GATE_MAX_MIN:-15}" RESEARCH_RUN_DIR="$D/out/gate" "${BIND[@]}" \
    bash benchmarks/hash/dither/sbench.sh "${BASE[@]}" --reps 0 --gate "${extra[@]}"
  rc=$?
  echo "prefill: gate pass exited $rc at $(date -u +%H:%M:%SZ)"
  if [ $rc = 0 ]; then
    echo "prefill: time pass ${BASE[*]} --reps ${REPS:-3} ${extra[*]}"
    RESEARCH_RUN_DIR="$D/out/time" "${BIND[@]}" bash benchmarks/hash/dither/sbench.sh "${BASE[@]}" --reps "${REPS:-3}" "${extra[@]}"
    rc=$?
  fi
else
  GATE_ARG=(--gate); [ "${GATE:-1}" = 0 ] && GATE_ARG=()
  RESEARCH_RUN_DIR="$D/out" "${BIND[@]}" bash benchmarks/hash/dither/sbench.sh "${BASE[@]}" --reps "${REPS:-1}" "${GATE_ARG[@]}" "${extra[@]}"
  rc=$?
fi
kill "$sampler" 2>/dev/null
echo "prefill: exited $rc at $(date -u +%H:%M:%SZ)"
exit $rc
