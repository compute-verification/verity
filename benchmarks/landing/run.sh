#!/usr/bin/env bash
# Mark to `main` (README.md): landing.py over its window (the 7 days before now, unless --days or --until, which this
# passes on, say otherwise). It reads the control pod (`research pods ssh vy-control-verity`), GitHub (`gh`) and the evidence
# store's labels, which an agent VM with the team's keys reaches and a CPU node does not (none of the three), so it runs as
# a local research run from a checkout there:
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
