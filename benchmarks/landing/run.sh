#!/usr/bin/env bash
# Mark to `main` (README.md): landing.py over its window (the 7 days before now, unless --days or --until, which this
# passes on, say otherwise). It reads the control pod (`research pods ssh vy-control-verity`), GitHub (`gh`) and the evidence
# store's labels, which no CPU node reaches, so it runs as a local research run on the control pod (its rows' `runs`), where
# the pod's log is read in place, or on an agent VM with the team's keys:
#   research run --project verity --cwd source -- bash benchmarks/landing/run.sh
# Exit: 0, every input read and the result written; 1, an input not read (no measurement, landing.json says which);
# 2, the arguments.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT" || exit 2
: "${RESEARCH_RUN_DIR:?run.sh is the command of a research run (its header)}"
# the shipped tree's queue code, not the runner's copy research run puts on PYTHONPATH
export PYTHONPATH=$ROOT/tools/research/src:$ROOT/benchmarks/harness:$ROOT/tools/config:$ROOT
exec python3 benchmarks/landing/landing.py --repo "$ROOT" --out "${OUT:-$RESEARCH_RUN_DIR/out}" "$@"
