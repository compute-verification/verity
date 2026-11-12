#!/usr/bin/env bash
# hidden_zk's runner: one job of the directory by its name, with the rest of the arguments and the environment handed to
# the job's script as they are. A job is a script beside this file, named without `.sh`. Each job is the command of a
# `research run` from a pushed commit on a node (cwd = the source tree); its script's header has the full command, its
# knobs and what it writes, and README.md's table has them side by side.
#
#   bash benchmarks/pouw/hidden_zk/run.sh JOB [ARGS...]
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
job=${1:-}
case "$job" in
  ""|run|*[!a-z0-9_]*) ;;
  *) if [ -f "$HERE/$job.sh" ]; then
       shift
       exec bash "$HERE/$job.sh" "$@"
     fi ;;
esac
names=$(cd "$HERE" && ls -- *.sh | sed -e 's/\.sh$//' -e '/^run$/d' | paste -sd ' ')
echo "usage: run.sh JOB [ARGS...]   JOB: one of ${names} (got '${job}')" >&2
exit 2
