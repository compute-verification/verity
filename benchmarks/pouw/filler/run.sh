#!/usr/bin/env bash
# The filler directory's runner: one job by its name, with the rest of the arguments and the environment handed to the
# job's script as they are. Each job is the command of a `research run` from a pushed commit, on a node with an
# RTX PRO 6000 (cwd = the source tree); its script's header has its knobs, and README.md's table has what each measures
# and writes.
#
#   bash benchmarks/pouw/filler/run.sh JOB [ARGS...]
#   research run --on MACHINE --project verity --source . --cwd source --env GPU_LEASE_WHO=LANE \
#     -- bash benchmarks/pouw/filler/run.sh score ARM
#
# JOB is `score` (score.py, the scorer: every argument is its own), `tpcomm` (comm/tpcomm.sh) or a script beside this
# file, named without `.sh`.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
JOBS="capfill_co_probe|capfill_fit_probe|capfill_fold_probe|capfill_fstep_probe|dcap|dfill|fill|flin|fold_probe|pfill_probe|psched|psched_probe|tile_probe"
job=${1:-}
case "$job" in
  score) shift; exec python3 "$HERE/score.py" "$@" ;;
  tpcomm) shift; exec bash "$HERE/comm/tpcomm.sh" "$@" ;;
esac
if [[ "$job" =~ ^($JOBS)$ ]]; then
  shift
  exec bash "$HERE/$job.sh" "$@"
fi
echo "usage: run.sh JOB [ARGS...]   JOB: score, tpcomm, ${JOBS//|/, } (got '${job}')" >&2
exit 2
