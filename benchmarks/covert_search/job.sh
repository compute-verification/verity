#!/usr/bin/env bash
# job.sh ISLAND START CLASS SEED [meter.py args...]: one candidate on the node, as launch.py runs it. vy-nebius-2 refuses
# `research run --gpus` (its GPUs go out through gpu-lease inside a run), so the meter takes one GPU here.
set -u
P=${COVERT_PYTHON:-/workspace/jobs/venv312/bin/python}
here=$(cd "$(dirname "$0")" && pwd)
export GPU_LEASE_WHO=${GPU_LEASE_WHO:-training:covert-search}
exec gpu-lease 1 --wait --max-min "${COVERT_LEASE_MIN:-25}" --mem-gb 48 -- "$P" "$here/meter.py" --island "$here/islands/$1" --start "$2" --class "$3" --seed "$4" --out "$OUT" "${@:5}"
