#!/usr/bin/env bash
# CIR-263 at the PR's head, after the suites: the adapter's ratchet gate against BASE (the main the branch merged), pins --stale,
# and the retired-terms tests, each to $OUT.
set -uo pipefail
: "${BASE:?BASE, the main the branch merged}"
date -u +"sync %FT%TZ"; git log -1 --format="head %H"
git cat-file -e "$BASE^{commit}" 2>/dev/null || git fetch -q origin main 2>/dev/null || true
git cat-file -e "$BASE^{commit}" 2>/dev/null || { echo "no commit $BASE in this tree"; exit 2; }
uv sync -q --all-packages --extra torch-cpu || exit 2

date -u +"gate %FT%TZ"
python3 adapters/vllm/tools/ratchet_gate.py "$BASE" HEAD --json > "$OUT/gate.json" 2> "$OUT/gate.err"
g=$?
echo "gate rc=$g"; head -c 2000 "$OUT/gate.json"; echo; tail -5 "$OUT/gate.err"

date -u +"pins %FT%TZ"
python3 tools/verity/check/pins.py --stale > "$OUT/pins.txt" 2>&1
p=$?
echo "pins --stale rc=$p"; cat "$OUT/pins.txt"

date -u +"retired %FT%TZ"
uv run pytest -q -p no:cacheprovider -rA -W always tests/test_retired_terms.py tests/lint/test_repo_retired_terms.py > "$OUT/retired.log" 2>&1
r=$?
echo "retired rc=$r"; grep -E "passed|failed|PASSED|FAILED|Warning|Bookkeeping" "$OUT/retired.log" | tail -20
date -u +"end %FT%TZ"
exit $(( g || p || r ))
