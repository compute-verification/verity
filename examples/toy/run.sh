#!/usr/bin/env bash
# `verity example toy [ARGS]`: the CPU toy (integrations/torch_loop, a two-layer W8A8 MLP) through every protocol and the
# wipe, composed through onsite: `verity bench all-protocols`, benchmarks/all_protocols/run.sh, with ARGS passed to its
# runner (--of-record, --windows N, --wired, --tie-tamper W, ...; that script's header lists them). It writes under $OUT, or
# $RESEARCH_RUN_DIR/out inside `research run`. The run of record is S1's, started as:
#   research run --on vy-nebius-cpu-4 --project verity --source . --cwd clone --mem-gb 96 --store-read -- \
#     bash examples/toy/run.sh --of-record --wired --windows 2 --tie-tamper 0
# The sampled-proofs roles run on their own is DEPLOYSP's toy_network example (#4928), a separate example.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
exec bash benchmarks/all_protocols/run.sh "$@"
