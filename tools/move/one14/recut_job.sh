#!/usr/bin/env bash
# tools/move/one14/recut_job.sh TIP: on a node, the ONE-14 move on TIP twice, as layout.py alone writes it (raw) and with
# fix_move.py's fixes (regen.sh), each certified by TIP's move_check.py and rerun once; both commits bundled into $OUT.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
tip=$(git rev-parse --verify "$1^{commit}") || exit 2
export GIT_AUTHOR_NAME="Cursor Agent" GIT_AUTHOR_EMAIL=cursoragent@cursor.com
export GIT_COMMITTER_NAME="Cursor Agent" GIT_COMMITTER_EMAIL=cursoragent@cursor.com
work=$(mktemp -d)
map="$work/map.toml"
cp "$here/../one14_map.toml" "$map"
echo "tip $tip"

rerun() {  # rerun the map on the worktree's commit and say whether it changes anything
  (cd "$1" && python3 tools/move/layout.py --map "$map" --no-lock --report-splits > "$OUT/$2.rerun.log" 2>&1
   echo "$2 rerun rc=$?"; git status --porcelain | tee "$OUT/$2.rerun.status" | wc -l | sed "s/^/$2 rerun changed paths: /"
   git reset -q --hard && git clean -qfd)
}

git worktree add -q --detach "$work/raw" "$tip"
(cd "$work/raw" && python3 tools/move/layout.py --map "$map" --no-lock --report-splits --report "$OUT/raw.report.json" \
   > "$OUT/raw.layout.log" 2>&1; echo "raw layout rc=$?"
 git add -A && git -c commit.gpgsign=false commit -q -m "ONE-14 move: layout.py's output alone" && git branch -f one14-raw HEAD
 echo "raw commit $(git rev-parse HEAD)"
 python3 tools/verity/check/move_check.py "$tip" HEAD --json > "$OUT/raw.move_check.json" 2> "$OUT/raw.move_check.err"
 echo "raw move_check rc=$?"; tail -5 "$OUT/raw.move_check.err")
rerun "$work/raw" raw

bash "$here/regen.sh" "$tip" "$work/fixed" > "$OUT/fixed.regen.log" 2>&1
echo "fixed regen rc=$?"; tail -8 "$OUT/fixed.regen.log"
cp "$work/fixed.layout.log" "$OUT/fixed.layout.log" 2>/dev/null; cp "$work/fixed.report.json" "$OUT/fixed.report.json" 2>/dev/null
(cd "$work/fixed" && git branch -f one14-move HEAD && echo "fixed commit $(git rev-parse HEAD)"
 python3 tools/verity/check/move_check.py "$tip" HEAD --json > "$OUT/fixed.move_check.json" 2> "$OUT/fixed.move_check.err"
 echo "fixed move_check rc=$?")
rerun "$work/fixed" fixed

git bundle create "$OUT/move.bundle" one14-raw one14-move "^$tip" && echo "bundle ok"
