#!/usr/bin/env bash
# The partial-root attack on k = 1 Sloth words, CPU only: completion against a root at 512-2048 bits, then γ at the
# built 1,536-bit point re-priced at the measured completion ratios. Writes out/partial_root.jsonl and
# out/partial_gamma.jsonl.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p out
WORDS=${WORDS:-300}
uv run --no-project --with gmpy2 --with python-flint python "$here/partial_root.py" "$WORDS" | tee out/partial_root.jsonl
fracs=$(python3 - out/partial_root.jsonl <<'EOF'
import json, sys
best = {}
for line in open(sys.argv[1]):
    d = json.loads(line)
    if d["reducer"] == "flint" and d["completed"] == d["words"]:
        best[d["field"]] = max(best.get(d["field"], 0), d["complete_over_root"])
print(" ".join(str(best[f]) for f in ("p1536", "p2048") if f in best), "0.2")
EOF
)
python3 "$here/partial_gamma.py" "$here/theory_model" $fracs | tee out/partial_gamma.jsonl
