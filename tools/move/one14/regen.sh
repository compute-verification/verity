#!/usr/bin/env bash
# tools/move/one14/regen.sh TIP [DIR]: the ONE-14 move on commit TIP, as one commit in a new detached worktree DIR (default
# a fresh temporary directory), and move_check's verdict on TIP..that commit (exit 0 certified, 1 refused).
# Run it from a checkout of a branch holding this directory and tools/move/one14_map.toml; it changes nothing there.
# The commit is layout.py's output with fix_move.py's three fixes; its parent is TIP.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
tip=$(git rev-parse --verify "$1^{commit}")
dir=${2:-$(mktemp -d)/one14}
map=$(mktemp --suffix=.toml)
cp "$here/../one14_map.toml" "$map"
git worktree add -q --detach "$dir" "$tip"
cd "$dir"
python3 tools/move/layout.py --map "$map" --no-lock --report-splits --report "$dir.report.json" > "$dir.layout.log"
python3 "$here/fix_move.py"
git -c commit.gpgsign=false commit -q -F "$here/move_message.txt"
echo "regen: move commit $(git rev-parse HEAD) on $tip in $dir"
python3 tools/verity/check/move_check.py "$tip" HEAD
