#!/usr/bin/env bash
# carry.sh [REMOTE_BRANCH]: run inside a lane's checkout, on its branch made before the protocol groups' rename. Carries
# the branch across the rename and the library move on REMOTE_BRANCH (default origin/one-circuit) with restack.py, then
# rewrites the branch's own lines that only hand.py reads, and leaves the result on a new local branch BRANCH-renamed (no
# push). The helpers and maps are REMOTE_BRANCH's own, so a branch made before this directory existed carries too.
set -euo pipefail
own=$(cd "$(dirname "$0")" && pwd)
cd "$(git rev-parse --show-toplevel)"
onto=${1:-origin/one-circuit}
branch=$(git branch --show-current)
[ -n "$branch" ] && [ -z "$(git status --porcelain)" ] || { echo "carry.sh: be on your branch with a clean worktree" >&2; exit 1; }
git fetch -q "${onto%%/*}" "${onto#*/}"
wt=$(mktemp -d)
trap 'git worktree remove --force "$wt"' EXIT
git worktree add -q --detach "$wt" "$onto"
here=$wt/tools/move/layout_names
[ -d "$here" ] || here=$own     # an onto branch made before this directory: this script's own
moves=()
for m in protocol_names_map library_datasets_map; do
  move=$(git log "$onto" -1 --format=%H --grep="^Layout-move: tools/move/$m.toml")
  [ -n "$move" ] || { [ $m = protocol_names_map ] && { echo "carry.sh: $onto holds no rename commit yet" >&2; exit 1; }; continue; }
  git show "$move:tools/move/$m.toml" | cmp -s - "$here/$m.toml" || { echo "carry.sh: $move ran another map than $here's" >&2; exit 1; }
  git merge-base --is-ancestor "$move" HEAD || moves+=("$move")
done
[ ${#moves[@]} -gt 0 ] || { echo "carry.sh: $branch already holds the rename"; exit 0; }
for move in "${moves[@]}"; do
  hand=$(git show -s --format=%B "$move" | sed -n 's/^Hand-paths: //p')
  { printf '\n["%s"]\nhand = [' "$move"; printf '"%s", ' $hand; printf ']\nreason = "hand.py after layout.py"\n'; } \
    | sed 's/, ]/]/' >> "$wt/tools/move/move_lines.toml"
done
flags=(); for move in "${moves[@]}"; do flags+=(--move-commit "$move"); done
out=$(python3 "$wt/tools/move/restack.py" "$branch" "${flags[@]}" --onto "$onto" --main "$onto" --no-pins --dry-run --json) || true
read -r status result < <(python3 -c 'import json,sys; r=json.loads(sys.argv[1])[0]; print(r["status"], r.get("result", ""))' "$out")
case $status in
  clean) ;;
  locks-stale) echo "carry.sh: your entries in lean-audit.json files were dropped; re-add them (restack.py's LOCKS_READD)" >&2 ;;
  *) echo "$out"; echo "carry.sh: restack.py stopped at $status: its report above names the step and the files" >&2; exit 1 ;;
esac
git checkout -q -b "$branch-renamed" "$result"
git diff --name-only "$onto" HEAD > "$wt/../own-$$.txt"
python3 "$here/hand.py" "$here/protocol_names_map.toml" --files "$wt/../own-$$.txt" > /dev/null
python3 tools/check/import_order.py > /dev/null || echo "carry.sh: tools/check/import_order.py asks for --grow: your branch's own imports" >&2
python3 -c 'import subprocess, sys; sys.path.insert(0, "tools/check"); import pins
[subprocess.run(p.shell(pins.ROOT), shell=True, check=True, cwd=pins.ROOT) for p in pins.PINS if p.key is None]'   # sha256sum pins: no Lean
if [ -n "$(git status --porcelain)" ]; then
  git commit -qam "Carry: the branch's own lines in the protocol groups' new names (hand.py, the import order's generator)"
fi
echo "carry.sh: $branch carried to $branch-renamed at $(git rev-parse HEAD); read it, then push it"
