#!/usr/bin/env bash
# One research run of the mutation harness over STEPS (comma list of spec.json step names); results in $OUT.
set -euo pipefail
STEPS=$1; JOBS=${2:-11}
D=benchmarks/door_mutants
uv sync -q --all-packages
base=$(mktemp -d); trap 'rm -rf "$base"' EXIT
python3 $D/mutate.py gen $D/spec.json "$OUT/mutants.jsonl"
export MUT_NOLAKE=1 MUT_BASE=$base
.venv/bin/python $D/mutate.py reach $D/spec.json "$OUT/reach.json" --jobs "$JOBS" --step "$STEPS"
.venv/bin/python $D/mutate.py run $D/spec.json "$OUT" --jobs "$JOBS" --step "$STEPS"
.venv/bin/python $D/mutate.py table "$OUT" | tee "$OUT/table.md"
