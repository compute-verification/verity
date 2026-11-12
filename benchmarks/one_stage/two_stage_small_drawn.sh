#!/usr/bin/env bash
# two_stage_small.sh on the greedy select in 20 units (V = 2560), then two_stage_drawn.py on its second stage's files:
# two members (a chunk's, and the combine's, whose ports read its writers' rows by wires and whose return reads round
# 1's rows) each restaged at its sampled proof blocks alone, one --zk session on each that samples every unit of the
# statement, and Lean's verdict on it under the verifier's own registered reads. The run's exit and records are
# two_stage_small.sh's; the restaged statements' verdicts are $OUT/after.log's DRAWN lines and $OUT/after/drawn.json,
# and machine.txt's "after rc" is 0 when both members held.
#   research run --on auto --cwd clone --cpus 16 --mem-gb 24 --max-min 120 -- bash benchmarks/one_stage/two_stage_small_drawn.sh
set -euo pipefail
SMALL_ARGS="--inline --units 20 --v 2560" SMALL_AFTER="benchmarks/one_stage/two_stage_drawn.py --members 0,19" \
  exec bash "$(dirname "$0")/two_stage_small.sh"
