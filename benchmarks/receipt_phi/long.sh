#!/usr/bin/env bash
# `sat_trails.py long` on duo2 (cwd = the source tree): the model check, then every start of T0 up the bounds,
# lowest pending bound first. As steps finish: $OUT/steps.jsonl (one line each), $OUT/levels.jsonl (one line each
# time every start of T0 is UNSAT at a new bound) and $OUT/proved.json (the current state). pysat comes from the sent
# wheel, unpacked into scratch. CPUS pins the solvers (vy-nebius-2's verity share: 48-95); under slot.py leave it
# unset, since slot.py pins. Everything runs at nice 19 and stops itself at JOB_MAX seconds.
#
#   research run --on vy-nebius-2 --project verity --campaign pouw --source . --cwd clone --timeout 43200 \
#     --send /abs/benchmarks/receipt_phi/long.sh --send /abs/python_sat-*.whl --env T0=0-255:10 --env CPUS=48-95 \
#     -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/long.sh"'
set -uo pipefail
O=${OUT:?}
D=benchmarks/receipt_phi
L=$(mktemp -d /tmp/receipt-long.XXXXXX); trap 'rm -rf "$L"' EXIT
python3 -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' \
  "$(ls "$RESEARCH_RUN_DIR"/inputs/python_sat-*.whl)" "$L" || exit 4
export PYTHONPATH=$L
pin=()
[ -n "${CPUS:-}" ] && pin=(taskset -c "$CPUS")
run() { "${pin[@]}" nice -n 19 "$@"; }
run python3 $D/sat_trails.py check > "$O/check.jsonl" 2>> "$O/errors.txt" || exit 5
run timeout "${JOB_MAX:-42600}" python3 $D/sat_trails.py long --kind "${KIND:-duo2}" --t0 "${T0:?}" \
  --bounds "${BOUNDS:-18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,47}" --k0 "${K0:-8}" --kmax "${KMAX:-16}" \
  --jobs "${JOBS:-8}" --out "$O" 2>> "$O/errors.txt"
rc=$?
echo "long rc $rc" >> "$O/errors.txt"
[ "$rc" = 124 ] && exit 0
exit "$rc"
