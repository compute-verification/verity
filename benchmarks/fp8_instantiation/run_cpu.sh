#!/usr/bin/env bash
# detection.py (#4) and rank.py (#9) on vy-nebius-cpu-1 in one of its proofs slots (cwd = the clone's root; the node's
# python3 has numpy, and Pearl-C's Python needs nothing else).  A4_PARTS=detection runs detection.py alone;
# A4_PARTS=fix runs the proposed fix rho' = max(rho, s/4) instead (detection.py --fix, rank.py --stats rho_s4), and
# A4_PARTS=low detection.py --low-r, A4_PARTS=cond detection.py --conditional.  research run passes no environment: set it inside the slot's command,
# `bash -c 'A4_PARTS=fix bash "$RESEARCH_RUN_DIR/inputs/run_cpu.sh"'` (the node refuses anything but slot.py after env).
#
#   research run --on vy-nebius-cpu-1 --project verity --campaign pouw --source . --cwd clone \
#     --send benchmarks/fp8_instantiation/run_cpu.sh -- env CHECK_SLOTS=/workspace/research/locks/proofs/slots \
#     python3 tools/verity/check/slot.py --wait-max 1800 -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/run_cpu.sh"'
set -uo pipefail
O=${RESEARCH_RUN_DIR:?}/out
mkdir -p "$O"
export PYTHONPATH=$PWD
SRC=benchmarks/fp8_instantiation
python3 -c 'import sys, numpy; print(sys.version.split()[0], numpy.__version__)' > "$O/python.txt" 2>&1
if [ "${A4_PARTS:-}" = cond ]; then
  python3 $SRC/detection.py --conditional --lines 200 --json "$O/conditional.json" > "$O/conditional_stdout.txt" 2>&1
  rc=$?
  echo "conditional $rc" | tee "$O/status.txt"
  exit $rc
fi
if [ "${A4_PARTS:-}" = low ]; then
  python3 $SRC/detection.py --low-r --draws 4000 --json "$O/low_r.json" > "$O/low_r_stdout.txt" 2>&1
  rc=$?
  echo "low_r $rc" | tee "$O/status.txt"
  exit $rc
fi
if [ "${A4_PARTS:-}" = fix ]; then
  python3 $SRC/detection.py --fix --json "$O/fix_detection.json" > "$O/fix_detection_stdout.txt" 2>&1 &
  det=$!
  python3 $SRC/rank.py --rows 512 --jobs 13 --stats rho_s4 --only dead_chunk,dead_chunk_wide,near_zero,outliers,gauss \
    --json "$O/fix_rank.json" > "$O/fix_rank_stdout.txt" 2>&1
  rk=$?
  wait $det
  dt=$?
  echo "fix_detection $dt fix_rank $rk" | tee "$O/status.txt"
  [ $dt = 0 ] && [ $rk = 0 ]
  exit
fi
python3 $SRC/detection.py --draws 4000 --json "$O/detection.json" > "$O/detection_stdout.txt" 2>&1 &
det=$!
rk=0
if [ "${A4_PARTS:-detection rank}" != detection ]; then
  python3 $SRC/rank.py --rows 512 --jobs 13 --json "$O/rank.json" > "$O/rank_stdout.txt" 2>&1
  rk=$?
fi
wait $det
dt=$?
echo "detection $dt rank $rk" | tee "$O/status.txt"
[ $dt = 0 ] && [ $rk = 0 ]
