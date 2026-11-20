#!/usr/bin/env bash
# One research run of the mutation harness over STEPS (comma list of spec.json step names); results in $OUT.
set -euo pipefail
STEPS=$1; JOBS=${2:-11}
D=benchmarks/door_mutants
uv sync -q --all-packages
if [ -n "${PREBUILD:-}" ]; then
  for p in verity/circuits/lean verity/protocols/proofs/flock/verifier verity/protocols/proofs/spot/verifier/lean; do
    bash tools/verity/lean/setup.sh $p > /dev/null
  done
  (cd verity/protocols/proofs/flock/verifier && lake build flock-lower-check > /dev/null)
  (cd verity/protocols/proofs/spot/verifier/lean && lake build > /dev/null)
fi
base=$(mktemp -d); trap 'rm -rf "$base"' EXIT
python3 $D/mutate.py gen $D/spec.json "$OUT/mutants.jsonl"
export MUT_NOLAKE=1 MUT_BASE=$base
.venv/bin/python $D/mutate.py reach $D/spec.json "$OUT/reach.json" --jobs "$JOBS" --step "$STEPS"
.venv/bin/python $D/mutate.py run $D/spec.json "$OUT" --jobs "$JOBS" --step "$STEPS"
.venv/bin/python $D/mutate.py table "$OUT" | tee "$OUT/table.md"
