#!/usr/bin/env bash
# `bash benchmarks/harness/steady_point.sh --spec SPEC --point POINT --identity IDENTITY`: one steady-state point
# (`python -m verity_bench.steady_node ARGS...`) as the research run `verity-bench run SPEC --direct NODE` launches, from
# the source tree (--cwd source), inside a GPU slot of the node, which gives it its cores.
# The dispatcher forwards the spec's env and its setup script as VERITY_BENCH_SETUP, which is sourced here, cwd the source
# tree. The setup builds the prover and makes the Python the point runs under, and exports both
# (VERITY_BENCH_STEADY_BIN, VERITY_BENCH_STEADY_PY), with whatever the prover, the stager and
# benchmarks/one_stage/server_rate.py need to run (the CUDA toolkit first on PATH, their PYTHONPATH and libraries).
set -euo pipefail
ROOT=$PWD
: "${VERITY_BENCH_SETUP:?a steady-state point runs in the environment its spec's setup leaves: launch it with verity-bench run SPEC --direct NODE}"
# shellcheck source=/dev/null
. "$ROOT/$VERITY_BENCH_SETUP"
: "${VERITY_BENCH_STEADY_PY:?$VERITY_BENCH_SETUP exported no VERITY_BENCH_STEADY_PY, the Python the point runs under}"
export PYTHONPATH=$ROOT/benchmarks/harness:$ROOT/verity/configuration/config${PYTHONPATH:+:$PYTHONPATH}
exec "$VERITY_BENCH_STEADY_PY" -m verity_bench.steady_node "$@"
