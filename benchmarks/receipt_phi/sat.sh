#!/usr/bin/env bash
# sat_trails.py on duo2 (cwd = the source tree): the model checked against trails.c's trails, then the open-end floor
# ladder over start words T0 (default all 256). pysat comes from the sent wheel. One JSON line per start in
# $OUT/floors.jsonl. Runs on vy-nebius-2's verity CPU share:
#
#   research run --on vy-nebius-2 --project verity --campaign pouw --source . --cwd clone \
#     --send /abs/benchmarks/receipt_phi/sat.sh --send /abs/python_sat-*.whl \
#     --env T0=0-127 -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/sat.sh"'
set -uo pipefail
O=${OUT:?}
D=benchmarks/receipt_phi
L=$(mktemp -d /tmp/receipt-sat.XXXXXX); trap 'rm -rf "$L"' EXIT
python3 -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' \
  "$(ls "$RESEARCH_RUN_DIR"/inputs/python_sat-*.whl)" "$L" || exit 4
export PYTHONPATH=$L
run() { taskset -c 48-95 nice -n 19 "$@"; }
run python3 $D/sat_trails.py check > "$O/check.jsonl" 2>> "$O/errors.txt" || exit 5
run timeout "${JOB_MAX:-4500}" python3 $D/sat_trails.py floors --kind "${KIND:-duo2}" --t0 "${T0:-0-255}" --k "${K:-8}" \
  --bounds "${BOUNDS:-13,16,20,24,28,32,36,40,44,47}" --conflicts "${CONFLICTS:-500000}" --jobs "${JOBS:-8}" \
  > "$O/floors.jsonl" 2>> "$O/errors.txt" || echo "floors rc $?" >> "$O/errors.txt"
