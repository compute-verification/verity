#!/usr/bin/env bash
# Pass 2: the pass-1 survivors listed in pass1/survivors.txt rerun with slow tests and lake (Lean built once in the
# clone, copied into each worker), against pass 1's reach maps in pass1/*reach.json. Results in $OUT/results2.jsonl.
set -euo pipefail
STEPS=$1; JOBS=${2:-8}
D=benchmarks/door_mutants
uv sync -q --all-packages
for p in verity/circuits/lean verity/protocols/proofs/flock/verifier verity/protocols/proofs/spot/verifier/lean; do
  bash tools/verity/lean/setup.sh $p > /dev/null
done
(cd verity/protocols/proofs/flock/verifier && lake build flock-lower-check > /dev/null)
(cd verity/protocols/proofs/spot/verifier/lean && lake build > /dev/null)
base=$(mktemp -d); trap 'rm -rf "$base"' EXIT
python3 $D/mutate.py gen $D/spec.json "$OUT/mutants.jsonl"
python3 - "$OUT/reach.json" $D/pass1/*reach.json <<'EOF'
import json, sys
m = {}
for p in sys.argv[2:]:
    m.update(json.load(open(p)))
json.dump(m, open(sys.argv[1], "w"))
EOF
export MUT_BASE=$base MUT_SLOW=1 MUT_IDS=$D/pass1/survivors.txt MUT_RESULTS=results2.jsonl MUT_BUDGET=2400
.venv/bin/python $D/mutate.py run $D/spec.json "$OUT" --jobs "$JOBS" --step "$STEPS"
