#!/usr/bin/env bash
# One research run of the Lean mutants (lean_mutants.json), after setting up Lean for the three packages.
set -euo pipefail
uv sync -q --all-packages
for p in verity/circuits/lean verity/protocols/proofs/flock/verifier verity/protocols/proofs/spot/verifier/lean; do
  bash tools/verity/lean/setup.sh $p > /dev/null
done
.venv/bin/python benchmarks/door_mutants/lean_mutants.py benchmarks/door_mutants/lean_mutants.json "$@"
