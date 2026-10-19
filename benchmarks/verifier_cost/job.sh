#!/usr/bin/env bash
# verifier_cost on a CPU node: build the NCI check (`nci-check`) and measure every config into $OUT/results.json
# (run.py measure). Run from a clone (the build writes .lake):
#   research run --as proofs --project verity --on vy-nebius-cpu-2 --source . --pool proofs --cwd clone -- \
#     bash benchmarks/verifier_cost/job.sh --unrolled-max 40000000
# Arguments go to run.py measure. Then, from a VM with the store's keys (a node's shell has none):
# `verity bench verifier_cost store RUN` and `verity bench verifier_cost plot ART --to PNG`.
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
uv sync -q --all-packages --extra torch-cpu
(cd verity/core/protocols/network/nci/lean && lake build nci-check) > "$O/build.log" 2>&1 || { tail -40 "$O/build.log" >&2; exit 1; }
uv run python benchmarks/verifier_cost/run.py measure --out "$O" "$@"
