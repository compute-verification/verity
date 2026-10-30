#!/usr/bin/env bash
# verifier_cost on a CPU node: build the verifier COMMAND measures (`measure`: the inference-only check, `nci-check`; `partition`:
# the one-stage pieces, `sampled-proofs-difftest`) and run `run.py COMMAND` into $OUT. Run from a clone (the build
# writes .lake):
#   research run --as proofs --project verity --on vy-nebius-cpu-2 --source . --pool proofs --cwd clone -- \
#     bash benchmarks/verifier_cost/job.sh measure --unrolled-max 40000000
# Arguments after COMMAND go to run.py COMMAND. Then, from a VM with the store's keys (a node's shell has none):
# `verity bench verifier_cost store RUN` and `verity bench verifier_cost plot ART --to PNG`.
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
cmd=${1:?usage: job.sh measure|partition [ARGS...]}
shift
case $cmd in
  measure) pkg=verity/core/protocols/network/nci/lean exe=nci-check ;;
  partition) pkg=verity/core/service/sampled_proofs exe=sampled-proofs-difftest ;;
  *) echo "job.sh: COMMAND is measure or partition, not $cmd" >&2; exit 2 ;;
esac
uv sync -q --all-packages --extra torch-cpu
(cd "$pkg" && lake build "$exe") > "$O/build.log" 2>&1 || { tail -40 "$O/build.log" >&2; exit 1; }
uv run python benchmarks/verifier_cost/run.py "$cmd" --out "$O" "$@"
