#!/usr/bin/env bash
# detection.py (#4) and rank.py (#9) on vy-nebius-cpu-1 in one of its proofs slots (cwd = the clone's root; the node's
# python3 has numpy, and Pearl-C's Python needs nothing else).  --env A4_PARTS=detection runs detection.py alone.
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
