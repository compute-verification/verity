#!/usr/bin/env bash
# The door predicates' Lean under mutation (admitted_lean_mutants.json, the ids given): each rebuilt with Spot's
# `#guard`s and spot-one-stage-zk, then the predicates' Lean-backed tests. Results in $OUT/admitted_lean_results.jsonl.
set -euo pipefail
uv sync -q --all-packages
for p in verity/circuits/lean verity/protocols/proofs/flock/verifier verity/protocols/proofs/spot/verifier/lean; do
  bash tools/verity/lean/setup.sh $p > /dev/null
done
S=tests/verity/protocols/proofs/spot/verifier
export MUT_RESULTS=admitted_lean_results.jsonl MUT_SPOT_EXES=spot-one-stage-zk
export MUT_TESTS=$S/test_door_form.py,$S/test_one_circuit_refusals.py,$S/test_members_run.py,$S/lean/test_restrict_block_circuit.py,$S/lean/test_restrict_difftest.py,$S/lean/test_block_front_difftest.py,$S/lean/test_second_wires_difftest.py,tests/verity/protocols/proofs/flock/prover/test_lean_lower.py,$S/test_one_circuit_chain.py,tests/benchmarks/one_stage/test_door_walls.py
.venv/bin/python benchmarks/door_mutants/lean_mutants.py benchmarks/door_mutants/admitted_lean_mutants.json "$@"
