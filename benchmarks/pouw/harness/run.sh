#!/usr/bin/env bash
# The harness's runner: one job of the directory by its name, started as it is with the rest of the arguments and the
# environment. `build` is native/build.sh and `pin_target` is jobs/pin_target.sh; `bench`, `characterize`, `verify`,
# `ledger`, `price_twins` and `ieee_pin` are the Python script of that name, under "$PY" where jobs/env.sh has set it and
# under python3 else. A job that needs a GPU is the command of a `research run`: README.md has each job's command, and
# the job's own header its arguments and what it writes.
#
#   bash benchmarks/pouw/harness/run.sh JOB [ARGS...]
#   bash benchmarks/pouw/harness/run.sh bench --plan     # each shape's operand sets, bursts and memory, without a GPU
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
job=${1:-}
case "$job" in
  build) shift; exec bash "$HERE/../../../verity/protocols/accounting/pouw/prover/kernels/harness/native/build.sh" "$@" ;;
  pin_target) shift; exec bash "$HERE/jobs/pin_target.sh" "$@" ;;
  bench|characterize|verify|ledger|price_twins|ieee_pin) shift; exec "${PY:-python3}" "$HERE/$job.py" "$@" ;;
esac
echo "usage: run.sh JOB [ARGS...]   JOB: build, bench, characterize, verify, ledger, price_twins, ieee_pin or" \
  "pin_target (got '${job}')" >&2
exit 2
