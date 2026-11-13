#!/usr/bin/env bash
# CIR-263 at the head the PR opens at: the suites the change from main touches, the adapter's ratchet gate and the retired-terms
# tests, each to $OUT.  BASE (required) is the main the branch merged; origin/main is used when the shipped tree has it and it
# names BASE.  Leaves TMPDIR as the run gives it.
set -uo pipefail
: "${BASE:?BASE, the main the branch merged}"
date -u +"sync %FT%TZ"; git log -1 --format="head %H"
base=$BASE
if git rev-parse -q --verify "origin/main^{commit}" >/dev/null && [ "$(git rev-parse origin/main)" = "$BASE" ]; then base=origin/main; fi
git cat-file -e "$BASE^{commit}" 2>/dev/null || git fetch -q origin main 2>/dev/null || true
git cat-file -e "$BASE^{commit}" 2>/dev/null || { echo "no commit $BASE in this tree"; exit 2; }
echo "base $base = $(git rev-parse "$base")"
uv sync -q --all-packages --extra torch-cpu || exit 2

date -u +"suites %FT%TZ"
uv run tools/verity/check/suites.py --changed "$base" --mem-gb 32 --json "$OUT/suites.json" --logs "$OUT/logs" > "$OUT/suites.log" 2>&1
s=$?
echo "suites rc=$s"; tail -40 "$OUT/suites.log"

date -u +"gate %FT%TZ"
python3 adapters/vllm/tools/ratchet_gate.py "$base" HEAD --json > "$OUT/gate.json" 2> "$OUT/gate.err"
g=$?
echo "gate rc=$g"; python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('failures', d.get('failures'))" "$OUT/gate.json"; tail -5 "$OUT/gate.err"

date -u +"pins %FT%TZ"
python3 tools/verity/check/pins.py --stale > "$OUT/pins.txt" 2>&1
p=$?
echo "pins --stale rc=$p"; cat "$OUT/pins.txt"

date -u +"retired %FT%TZ"
uv run pytest -q -p no:cacheprovider -rA -W always tests/test_retired_terms.py tests/lint/test_repo_retired_terms.py > "$OUT/retired.log" 2>&1
r=$?
echo "retired rc=$r"; grep -E "passed|failed|PASSED|FAILED|Warning|Bookkeeping" "$OUT/retired.log" | tail -20
date -u +"end %FT%TZ"
exit $(( s || g || p || r ))
