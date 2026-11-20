#!/usr/bin/env bash
# Which of the Python twin's refusals the drawn-blocks difference test (Lean's spot-one-stage-zk restrict against
# Python's door, test_door_walls.py) compares: every Python mutant of run_door2_py.sh against that one test.
set -euo pipefail
JOBS=${1:-10}
D=benchmarks/door_mutants
uv sync -q --all-packages
for p in verity/circuits/lean verity/protocols/proofs/flock/verifier verity/protocols/proofs/spot/verifier/lean; do
  bash tools/verity/lean/setup.sh $p > /dev/null
done
(cd verity/protocols/proofs/flock/verifier && lake build flock-lower-check > /dev/null)
(cd verity/protocols/proofs/spot/verifier/lean && lake build Spot spot-one-stage-zk > /dev/null)
T=$(python3 -c "import json;print(json.load(open('$D/spec_door2_diff.json'))['steps'][0]['slow'][0])")
.venv/bin/python -m pytest -q -rs -p no:cacheprovider "$T" | tail -3
base=$(mktemp -d); trap 'rm -rf "$base"' EXIT
python3 $D/mutate.py gen $D/spec_door2_diff.json "$OUT/mutants.jsonl"
python3 - "$OUT/mutants.jsonl" <<'PY'
import json, sys
f = "verity/protocols/proofs/spot/verifier/one_stage/one_circuit.py"
src = open(f).read()
a = src.index("    formed(key)\n") + 4
m = {"step": "door2", "file": f, "func": "lower", "line": src[:a].count("\n") + 1, "op": "call", "desc": "formed(key) not called",
     "a": a, "b": a + len("formed(key)"), "text": "pass", "orig": "formed(key)", "id": "door2-formed"}
open(sys.argv[1], "a").write(json.dumps(m) + "\n")
PY
echo '{}' > "$OUT/reach.json"
export MUT_BASE=$base MUT_SLOW=1 MUT_RESULTS=results2.jsonl MUT_BUDGET=1200
.venv/bin/python $D/mutate.py run $D/spec_door2_diff.json "$OUT" --jobs "$JOBS" --step door2
