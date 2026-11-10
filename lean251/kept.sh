#!/usr/bin/env bash
# LEAN-258: one `restack.py` as an owner runs it on the day, in a run's clone on a node, with the restack.py of commit PR:
#   kept.sh PR BRANCH MAIN NAME          BRANCH across every move MAIN has and it lacks (--dry-run --json)
#   kept.sh PR MOVE   MOVE NAME check    only the check of MOVE (the move commit as its own branch: up-to-date after the check)
# `--kept` is passed when PR's restack.py has it. A sixth argument, a sha256: the kept files which the restack.py of that
# digest wrote are removed at the end (a head of the PR that kept by default: its suites' tests left their checks in the
# node's cache, and nothing reads them again).
# Writes $OUT/kept-NAME.json (restack.py's), kept-NAME.stderr, kept-NAME.txt (seconds, the result's tree, the kept files).
set -u
pr=$1 branch=$2 main=$3 name=$4 mode=${5:-restack} gone=${6:-}
export GIT_AUTHOR_NAME=lean GIT_AUTHOR_EMAIL=lean@verity.invalid GIT_COMMITTER_NAME=lean GIT_COMMITTER_EMAIL=lean@verity.invalid
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
git cat-file blob "$pr:tools/move/restack.py" > "$T/restack.py" || exit 2
cat > "$T/gone.py" <<'G'
import json, os, sys
for name in sorted(os.listdir(sys.argv[1])) if os.path.isdir(sys.argv[1]) else []:
    path = os.path.join(sys.argv[1], name)
    try:
        mine = json.load(open(path))["key"]["restack"] == sys.argv[2]
    except (OSError, ValueError, KeyError, TypeError):
        mine = False
    if mine:
        os.unlink(path)
        print(f"removed {path} (written by the restack.py of sha256 {sys.argv[2][:16]})")
G
kept=$HOME/.cache/verity-check/restack-generated
flag=()
grep -q -- '"--kept"' "$T/restack.py" && flag=(--kept)
{
  echo "host=$(hostname) restack.py of $pr sha256=$(sha256sum "$T/restack.py" | cut -c1-16) flag=${flag[*]:-none}"
  echo "kept before: $(ls "$kept" 2>/dev/null | wc -l) file(s) in $kept -> $(readlink -f "$kept")"
} | tee "$OUT/kept-$name.txt"
s=$(date +%s)
if [ "$mode" = check ]; then
  python3 "$T/restack.py" "$branch" --move-commit "$main" --dry-run --json --no-pins ${flag[@]+"${flag[@]}"} > "$OUT/kept-$name.json" 2> "$OUT/kept-$name.stderr"
else
  python3 "$T/restack.py" "$branch" --main "$main" --dry-run --json ${flag[@]+"${flag[@]}"} > "$OUT/kept-$name.json" 2> "$OUT/kept-$name.stderr"
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
  [ -n "$gone" ] && python3 "$T/gone.py" "$kept" "$gone"
  echo "kept at the end: $(ls "$kept" 2>/dev/null | wc -l) file(s)"
} | tee -a "$OUT/kept-$name.txt"
exit 0
