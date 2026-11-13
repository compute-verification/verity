#!/usr/bin/env bash
# The rate of `halves_split` (scratch/cir264/halves_rate.py) as one CPU job, from the root of the tree it is committed in:
#   research run --on auto --project verity --source TREE --cwd source --cpus 16 --mem-gb 32 --timeout 45m \
#     [--env MINUTES=26] [--env WORKERS=16] -- bash scratch/cir264/halves_rate.sh
# It keeps $OUT/halves_rate/halves_rate.json (every sum the report is printed from) and halves_rate_positions.jsonl (each
# draft row whose halves' sums differ).  Its scratch is a mktemp directory, removed when it exits.
set -euo pipefail
: "${OUT:=$RESEARCH_RUN_DIR/out}"
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
date -u +"start %Y-%m-%dT%H:%M:%SZ on $(hostname), tree $(git rev-parse HEAD 2> /dev/null || basename "$PWD"), $(nproc) cores"
uv venv -q --python 3.12 "$d/venv"
uv pip install -q --python "$d/venv/bin/python" numpy==2.5.4
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
rc=0
"$d/venv/bin/python" scratch/cir264/halves_rate.py --out "$OUT/halves_rate" --minutes "${MINUTES:-26}" \
  --workers "${WORKERS:-16}" ${ARGS:-} || rc=$?
date -u +"end %Y-%m-%dT%H:%M:%SZ exit $rc"
exit "$rc"
