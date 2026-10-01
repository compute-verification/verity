#!/usr/bin/env bash
# check.sh: one compact status block for the 20-min console check.
set -u
W=$HOME/console-overnight
STORE=/cursor/stores/bc-7f347b4b-6175-4b6e-84c6-731add2f8589
echo "== $(date -u +%H:%MZ) / $(TZ=America/Los_Angeles date +%-I:%M\ %p) PDT"

pid=$(pgrep -xf "bash $W/loop.sh" | head -1)
# the VM can be suspended while idle, which stalls the loop's sleep: end the sleep so a pass runs now
s=$([[ -n "$pid" ]] && pgrep -P "$pid" -x sleep)
[[ -n "$s" ]] && kill "$s" && sleep 10
echo "loop: ${pid:-DEAD}"; tail -n 3 "$W/loop.log"

cd /workspace
uv run --quiet --package research research pods ssh vy-nebius-1 -- '
cd /workspace/usage/overnight && for f in *.json; do printf "%s %s %s\n" "$f" "$(date -u -r "$f" +%H:%MZ)" "$(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(d.get(\"updated_utc\"),len(d.get(\"numbers\",[])),\"rows\")" "$f" 2>&1)"; done
journalctl -u verity-console.service --no-pager -n 1 -o cat | python3 -c "import json,sys;l=sys.stdin.read();i=l.find(\"{\");d=json.loads(l[i:]) if i>=0 else {};print(\"publisher\",d.get(\"at\"),d.get(\"published\"),\"overnight_files\",d.get(\"overnight_files\"),\"errors\",d.get(\"errors\"))"
' 2>&1

awk '/^## Morning set/{f=1} /^## Overnight set/{f=0} f' "$STORE/docs/goals.md" > "$W/overnight-set.now.md"
if cmp -s "$W/overnight-set.now.md" "$W/overnight-set.last.md"; then echo "morning set: unchanged"
else echo "morning set: CHANGED"; diff "$W/overnight-set.last.md" "$W/overnight-set.now.md" | cut -c1-400; fi

cd "$HOME/.research/notes"
before=$(git rev-parse HEAD)
git pull -q --rebase 2>&1 | tail -1
git diff --name-only --diff-filter=A "$before" HEAD -- lanes/console lanes/coordinator 2>/dev/null | sed 's/^/new note: /'
