#!/usr/bin/env bash
# Dither's served prefill row on vy-nebius-2: sbench.sh --prefill --serial --t $T (2048) with exact, exact-fused and $ROW,
# the decode row's units and table builds (HB_KC_FLAGS, HB_KC_H1C_FLAGS) and window flags (DITHER_SO_FUSED: a step's
# units one h1um launch), except h1um's L2 prefetch: L2PF (0) is H1U_L2PF.  At prefill's m = 2048 every A and B line a
# unit reads is L2's already, and the prefetches cost h1um 2.3 % (unitsbench --multi, step_multi 397.6 -> 388.6 ms;
# at decode's m = 32 6.425 -> 6.414 ms).  KCX and KHX append to the two builds' flags (a KCX -DNAME=V in place of the
# base's -DNAME), EXTRA to sbench.py's arguments.
# A gate alone (REPS=0), a timed row alone (GATE=0 REPS=3), or both in one run (SPLIT=1); sbench.sh builds its
# libraries with nvcc before each lease:
#   research run --on vy-nebius-2 --mem-gb 8 --project verity --source . --cwd source --env GPU_LEASE_WHO=LANE \
#     --env SBENCH_ON=4,5,6,7 --env NUMA=0 --env GATE=0 --env REPS=3 -- bash benchmarks/hash/dither/prefill.sh
# SPLIT=1 runs two passes over the same arms and flags: out/gate, every arm gated and none timed, under an untimed lease
# of GATE_MAX_MIN (15) minutes asked at once, so it ends before a clean window opens; then out/time, every arm timed REPS
# (3) times and none gated, under the lease SBENCH_LEASE_AT and GPU_LEASE_TIMED ask for (the window's).  Without it, one
# pass into out/ (--gate unless GATE=0, REPS 1).  A gated pass's lease caps host memory at GATE_MEM_GB (192: the gate's
# captures sit in host memory and its check hashes them in DITHER_CHECK_WORKERS (8) processes; at 128 the kernel killed
# it, r20261024-053503-9e7d), a timed-only one at TIME_MEM_GB (64), not
# gpu-lease's per-GPU default; each summary.json's host_mem_peak_gib is what its lease used.  NUMA=n binds every process
# to node n's CPUs and memory (n2's GPUs 4-7 are node 1's, but merge trains may hold node 1's cores 64-119, so timed rows
# take node 0); out/load.csv samples the node's load every 10 s.  A J10-R row (--env DITHER_SO_UNITS=j10r --env
# J10R_SHA=559bc0aa69) takes DITHER_SO_RDEAD=0: its units are each computed in full ($DITHER_UNITS_SKIP_DEAD unset), so
# dither_so refuses the zero-receipt rows of 1, and at m = T no row block is dead.
# J10RC=1 is the J10-RC row (served step 341.0 ms at T = 2048 on the RTX, gamma 5.37 over exact-fused,
# r20261025-104529-8fc1): that J10-R row with its units as kh_units_multi_j10rc (DITHER_SO_J10RC_MIN_T 512) in 8-warp
# blocks over 2 x 4 patches of units, each row block's tail digits once a call, on WarpSmemX (KCX's JRY, JRX, WARPS,
# MINB and JSWZQ), the z tables without yu on uncaptured steps (DITHER_YCMP) and prefill's GEMM tiles (DITHER_MM_CFG,
# dmmsweep.py --grid prefill r20261024-182514-971d), its units' stream at the serving stream's priority
# (DITHER_SO_UNITS_PRIO -1: at the default 0 the forward's blocks take the SMs first and the trees wait 154 ms a step on
# the units; at -1, 369.4 -> 354.7 ms, and at -2 and -3 the forward waits instead, 389.6 ms; r20261024-225336-030b),
# and its units on 6 of every 8 SMs (DITHER_SO_UNITS_SMQ 6/8: h1umjc_sm, each CTA taking 8 neighbouring units under
# KCX's JCTA, so the forward keeps the rest whole; 8/8 361.5, 7/8 348.8, 6/8 340.5, 5/8 343.3, 4/8 397.6 ms,
# r20261025-000337-a414).  The environment's J10R_SHA, DITHER_SO_J10RC_MIN_T, DITHER_MM_CFG, DITHER_SO_UNITS_PRIO and
# DITHER_SO_UNITS_SMQ win, and a KCX of its own goes after these flags.
set -uo pipefail
if [ "${J10RC:-0}" = 1 ]; then
  export DITHER_SO_UNITS=j10r J10R_SHA=${J10R_SHA:-559bc0aa69} DITHER_SO_RDEAD=0 DITHER_YCMP=1
  export DITHER_SO_J10RC_MIN_T=${DITHER_SO_J10RC_MIN_T:-512} DITHER_SO_UNITS_PRIO=${DITHER_SO_UNITS_PRIO:--1}
  export DITHER_SO_UNITS_SMQ=${DITHER_SO_UNITS_SMQ:-6/8}
  export DITHER_MM_CFG=${DITHER_MM_CFG:-6144,4096,2048=128x128x64:w4:s4:k1:t1+4096,4096,2048=128x128x64:w4:s4:k1:t1+28672,4096,2048=128x64x64:w4:s3:k1:t1:g4+4096,14336,2048=128x128x64:w4:s4:k1:t1:g8}
  KCX="-DH1U_JRC=1 -DH1U_JSWZ=32 -DH1U_JRY=1 -DH1U_JRX=1 -DH1U_WARPS=8 -DH1U_MINB=1 -DH1U_JSWZQ=2 -DH1U_JCTA=1 ${KCX:-}"
fi
D=${RESEARCH_RUN_DIR:?}
mkdir -p "$D/out"
# FlashInfer's JIT cache is a private copy (its cuda_home the node's /usr/local/cuda-13.0, so nothing rebuilds): a timed
# lease's Quiet stops other runs' ninja, and one stopped holding the shared cache's sampling.lock hung a pass for good;
# it goes where sbench.sh's scratch does (SBENCH_SCRATCH_ROOT, /workspace/pouw, else the system's temporary directory)
FI=$(mktemp -d "${SBENCH_SCRATCH_ROOT:-/workspace/pouw}/tmp-prefill-fi.XXXXXX" 2>/dev/null) ||
  { [ -z "${SBENCH_SCRATCH_ROOT:-}" ] && FI=$(mktemp -d -t tmp-prefill-fi.XXXXXX); } || { echo "prefill: NO_SCRATCH"; exit 2; }
trap 'rm -rf "$FI"' EXIT
mkdir -p "$FI/.cache" && cp -a "$HOME/.cache/flashinfer" "$FI/.cache/" 2>/dev/null
export FLASHINFER_WORKSPACE_BASE=$FI CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
KC=""
for f in -DH1U_L2HINT=128 -DH1U_WARPS=2 -DH1U_EVF=1 -DH1S_EVL=1 -DH1U_EVFU=0 -DH1K_CPA=3 -DH1U_PF=1 -DH1U_NB2=1 -DH1U_L2PF=${L2PF:-0}; do
  case " ${KCX:-} " in *" ${f%%=*}="*) ;; *) KC+="$f " ;; esac
done
export HB_KC_FLAGS="$KC${KCX:-}"
export HB_KC_H1C_FLAGS="-DH1C_HB=1 -DH1C_MK=1 -DH1C_MID=1 -DH1C_DOMSEG=1 -DH1C_LVF=2 -DH1C_LMINB=4 -DH1C_APW=1 -DH1C_SEGMID=1 ${KHX:-}"
export DITHER_SO_FUSED=1 DITHER_ZDEAD=1 DITHER_SO_RDEAD=${DITHER_SO_RDEAD:-1} DITHER_A_STEP=1 DITHER_SO_COIN_ASYNC=2 DITHER_SO_PRIO=1 DITHER_ZY1_KC=1
export DITHER_SPILL_MAX_GB=${DITHER_SPILL_MAX_GB:-96} SBENCH_MAX_MIN=${SBENCH_MAX_MIN:-20}
export DITHER_CHECK_WORKERS=${DITHER_CHECK_WORKERS:-8}
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
  env -u SBENCH_LEASE_AT -u GPU_LEASE_TIMED SBENCH_MAX_MIN="${GATE_MAX_MIN:-15}" RESEARCH_RUN_DIR="$D/out/gate" \
    GPU_LEASE_MEM_GB_PER_GPU="${GATE_MEM_GB:-192}" "${BIND[@]}" \
    bash benchmarks/hash/dither/sbench.sh "${BASE[@]}" --reps 0 --gate "${extra[@]}"
  rc=$?
  echo "prefill: gate pass exited $rc at $(date -u +%H:%M:%SZ)"
  if [ $rc = 0 ]; then
    echo "prefill: time pass ${BASE[*]} --reps ${REPS:-3} ${extra[*]}"
    RESEARCH_RUN_DIR="$D/out/time" GPU_LEASE_MEM_GB_PER_GPU="${TIME_MEM_GB:-64}" "${BIND[@]}" \
      bash benchmarks/hash/dither/sbench.sh "${BASE[@]}" --reps "${REPS:-3}" "${extra[@]}"
    rc=$?
  fi
else
  GATE_ARG=(--gate); MEM=${GATE_MEM_GB:-192}
  [ "${GATE:-1}" = 0 ] && { GATE_ARG=(); MEM=${TIME_MEM_GB:-64}; }
  RESEARCH_RUN_DIR="$D/out" GPU_LEASE_MEM_GB_PER_GPU="$MEM" "${BIND[@]}" bash benchmarks/hash/dither/sbench.sh "${BASE[@]}" --reps "${REPS:-1}" "${GATE_ARG[@]}" "${extra[@]}"
  rc=$?
fi
kill "$sampler" 2>/dev/null
echo "prefill: exited $rc at $(date -u +%H:%M:%SZ)"
exit $rc
