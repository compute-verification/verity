#!/bin/bash
# usage (in a research run, --cwd clone of the tip): piece_job.sh MAP
# The suites the map's change touches, with the map and on the tip, and the retired-terms lint with the map.
set -o pipefail
MAP=$RESEARCH_RUN_DIR/inputs/$1
python3 tools/move/rename.py --map "$MAP" > "$OUT/rename.log" 2>&1 || { echo "rename failed"; tail -20 "$OUT/rename.log"; exit 1; }
git diff --cached --stat | tail -1
uv run tools/verity/check/suites.py --quick --changed HEAD --json "$OUT/after.json" --logs "$OUT/logs_after" > "$OUT/after.log" 2>&1
echo "after rc $?"
python3 -m pytest tests/lint/test_repo_retired_terms.py -q -p no:cacheprovider > "$OUT/lint.log" 2>&1
echo "lint rc $?"; tail -2 "$OUT/lint.log"
SUITES=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(' '.join(sorted(d if isinstance(d, dict) else [x['name'] for x in d])))" "$OUT/after.json")
echo "suites: $SUITES"
git reset -q --hard && git clean -fdq
uv run tools/verity/check/suites.py --quick $SUITES --json "$OUT/before.json" --logs "$OUT/logs_before" > "$OUT/before.log" 2>&1
echo "before rc $?"
exit 0
