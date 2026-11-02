#!/usr/bin/env bash
# leased.sh PYTHON ARGS...: train.py (or LEASED_SCRIPT, e.g. speedrun.py, which keeps the same exit codes) as a chain of
# preemptible one-GPU leases (gpu-lease), each under the 30-minute lease rule, until the run completes. A chunk's end (3), a
# lease's time limit (124) and a preemption (143) resume from the run's checkpoint; a refusal (75: a booked window ahead) asks
# again a minute later; any other exit is the script's.
set -u
here=$(cd "$(dirname "$0")" && pwd)
py=$1
shift
while true; do
  gpu-lease 1 --wait --preemptible --max-min 30 --mem-gb "${LEASE_MEM_GB:-64}" -- "$py" "$here/${LEASED_SCRIPT:-train.py}" "$@" --chunk-minutes 26
  rc=$?
  case $rc in
    3 | 124 | 143) ;;
    75) sleep 60 ;;
    *) exit $rc ;;
  esac
done
