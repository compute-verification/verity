#!/usr/bin/env bash
# LEAN-258: one `restack.py` as an owner runs it on the day, in a run's clone on a node, with the restack.py of commit PR:
#   kept.sh PR BRANCH MAIN NAME          BRANCH across every move MAIN has and it lacks (--dry-run --json)
#   kept.sh PR MOVE   MOVE NAME check    only the check of MOVE (the move commit as its own branch: up-to-date after the check)
# Writes $OUT/kept-NAME.json (restack.py's), kept-NAME.stderr, kept-NAME.txt (seconds, the result's tree, the kept files).
set -u
pr=$1 branch=$2 main=$3 name=$4 mode=${5:-restack}
export GIT_AUTHOR_NAME=lean GIT_AUTHOR_EMAIL=lean@verity.invalid GIT_COMMITTER_NAME=lean GIT_COMMITTER_EMAIL=lean@verity.invalid
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
git cat-file blob "$pr:tools/move/restack.py" > "$T/restack.py" || exit 2
kept=$HOME/.cache/verity-check/restack-generated
{
  echo "host=$(hostname) restack.py of $pr sha256=$(sha256sum "$T/restack.py" | cut -c1-16)"
  echo "kept before: $(ls "$kept" 2>/dev/null | wc -l) file(s) in $kept -> $(readlink -f "$kept")"
} | tee "$OUT/kept-$name.txt"
s=$(date +%s)
if [ "$mode" = check ]; then
  python3 "$T/restack.py" "$branch" --move-commit "$main" --dry-run --json --no-pins > "$OUT/kept-$name.json" 2> "$OUT/kept-$name.stderr"
else
  python3 "$T/restack.py" "$branch" --main "$main" --dry-run --json > "$OUT/kept-$name.json" 2> "$OUT/kept-$name.stderr"
fi
rc=$?
e=$(date +%s)
{
  echo "restack.py rc=$rc seconds=$((e - s)) minutes=$(python3 -c "print(round(($e - $s) / 60, 2))")"
  python3 - "$OUT/kept-$name.json" <<'E'
import json, subprocess, sys
for r in json.load(open(sys.argv[1])):
    tree = subprocess.run(["git", "rev-parse", f"{r['result']}^{{tree}}"], capture_output=True, text=True).stdout.strip() if r.get("result") else None
    print(f"status={r['status']} steps={[s['step'] for s in r.get('steps', [])]} result={r.get('result')} tree={tree} plain={r['plain']['status']}")
E
  echo "stderr (kept lines):"
  grep -n "restack.py:" "$OUT/kept-$name.stderr" | cut -c1-400
  echo "kept after: $(ls "$kept" 2>/dev/null | wc -l) file(s)"
  for f in "$kept"/*.json; do [ -f "$f" ] && { echo "== $f"; cat "$f"; echo; }; done
} | tee -a "$OUT/kept-$name.txt"
exit 0
