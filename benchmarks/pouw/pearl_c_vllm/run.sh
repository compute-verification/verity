#!/usr/bin/env bash
# pearl_c_vllm's runner: one job of the directory by its name, with the rest of the arguments and the environment handed
# to the job's script as they are. Each job is the command of a `research run`; its script's header has the full command,
# the machine it needs and what it writes (README.md's table has them side by side).
#
#   bash benchmarks/pouw/pearl_c_vllm/run.sh JOB [ARGS...]
#   JOB: setup, window, verify, served_passes, served_inputs, served_step, served_profile, noise_sweep or forming_domain
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
job=${1:-}
case "$job" in setup|window|verify|served_passes|served_inputs|served_step|served_profile|noise_sweep|forming_domain)
  shift
  exec bash "$HERE/$job.sh" "$@" ;;
esac
echo "usage: run.sh JOB [ARGS...]   JOB: setup, window, verify, served_passes, served_inputs, served_step," \
  "served_profile, noise_sweep or forming_domain (got '${job}')" >&2
exit 2
