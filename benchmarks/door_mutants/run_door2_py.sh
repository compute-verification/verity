#!/usr/bin/env bash
# The door predicates' Python twin under mutation: each refusal of one_circuit.admitted, template_form, admitted_block
# and lower made a no-op, and lower's call of `formed` removed; each mutant run against the test files that reach it,
# slow ones included, plus the predicates' own tests, Lean built once. Results in $OUT/results2.jsonl.
set -euo pipefail
JOBS=${1:-10}
D=benchmarks/door_mutants
uv sync -q --all-packages
for p in verity/circuits/lean verity/protocols/proofs/flock/verifier verity/protocols/proofs/spot/verifier/lean; do
  bash tools/verity/lean/setup.sh $p > /dev/null
done
(cd verity/protocols/proofs/flock/verifier && lake build flock-lower-check > /dev/null)
(cd verity/protocols/proofs/spot/verifier/lean && lake build Spot spot-one-stage-zk > /dev/null)
base=$(mktemp -d); trap 'rm -rf "$base"' EXIT
python3 $D/mutate.py gen $D/spec_door2.json "$OUT/mutants.jsonl"
python3 - "$OUT/mutants.jsonl" <<'PY'
import json, sys
f = "verity/protocols/proofs/spot/verifier/one_stage/one_circuit.py"
src = open(f).read()
old = "    formed(key)\n"
assert src.count(old) == 1
a = src.index(old) + 4
m = {"step": "door2", "file": f, "func": "lower", "line": src[:a].count("\n") + 1, "op": "call", "desc": "formed(key) not called",
     "a": a, "b": a + len("formed(key)"), "text": "pass", "orig": "formed(key)", "id": "door2-formed"}
open(sys.argv[1], "a").write(json.dumps(m) + "\n")
PY
export MUT_BASE=$base
.venv/bin/python $D/mutate.py reach $D/spec_door2.json "$OUT/reach.json" --jobs "$JOBS" --step door2
export MUT_SLOW=1 MUT_RESULTS=results2.jsonl MUT_BUDGET=2400
.venv/bin/python $D/mutate.py run $D/spec_door2.json "$OUT" --jobs "$JOBS" --step door2
