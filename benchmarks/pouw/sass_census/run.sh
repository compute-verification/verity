#!/usr/bin/env bash
# sass_census's runner: one job of the census by its name, with the rest of the arguments and the environment handed to
# the job's script as they are. Each job is the command of a `research run`; its script's header has the full command,
# the machine it needs and what it writes (README.md's table has them side by side).
#
#   bash benchmarks/pouw/sass_census/run.sh JOB [ARGS...]     # JOB: dump, rates, lowering, multimem or eps_counts
#   research run --on MACHINE --cwd source -- bash benchmarks/pouw/sass_census/run.sh dump [ROOT ...]
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
job=${1:-}
case "$job" in dump|rates|lowering|multimem|eps_counts)
  shift
  exec bash "$HERE/$job.sh" "$@" ;;
esac
echo "usage: run.sh JOB [ARGS...]   JOB: dump, rates, lowering, multimem or eps_counts (got '${job}')" >&2
exit 2
