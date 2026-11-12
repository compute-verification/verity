#!/usr/bin/env bash
# two_stage_small.sh on the greedy select in 20 units, each member's statement staged at its sampled proof blocks alone
# (two_stage.py second --sampled-only): its tables hold only the rows those blocks read, its session's sample is every
# unit of it, and round 2 holds it as a statement of only its part's sampled proof blocks. Exit 0 as for the class staged
# at every slot: every Lean verdict accepted, every negative refused, round 2 accepting the statements, and the outcome
# refused by the two-stage form alone.
#   research run --on auto --cwd clone --mem-gb 24 --max-min 120 -- bash benchmarks/one_stage/two_stage_small_sampled.sh
SMALL_ARGS="--inline --units 20 --v 2560" SMALL_SECOND="--sampled-only" exec bash "$(dirname "$0")/two_stage_small.sh"
