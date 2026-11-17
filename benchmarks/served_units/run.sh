#!/usr/bin/env bash
# verity bench served_units [UNIT ...]: job.sh on each UNIT (stage.py's UNITS type, TYPE[:S=V,...]; default
# fp8-coord:K=6144) at SU_N inputs (default 1), each unit's summary line last; exit 1 unless every unit is as expected.
# verity bench served_units --cover COVER [ARGS]: cover_count.py --record on a stored cover record (an art: id), held to
# ARGS' card (default GLM-5.3's).
set -uo pipefail
if [ "${1:-}" = --cover ]; then
  cover=${2:?--cover names a cover record: a cover.json or an art: id}
  shift 2
  [ $# -gt 0 ] || set -- --card benchmarks/sampling_plan/glm53_blocks.json
  exec python3 benchmarks/served_units/cover_count.py "$cover" --record "$@"
fi
export RESEARCH_RUN_DIR=${RESEARCH_RUN_DIR:-$PWD} SU_UNITS=${*:-${SU_UNITS:-fp8-coord:K=6144}} SU_N=${SU_N:-1}
export SU_MEM_GB=${SU_MEM_GB:-32}
bash benchmarks/served_units/job.sh || exit
s=$(python3 benchmarks/served_units/summarize.py "$RESEARCH_RUN_DIR/out") || exit
python3 benchmarks/served_units/record.py "$RESEARCH_RUN_DIR/out" || exit
echo "$s"
[[ -n $s && $s != *'"as_expected": false'* ]]
