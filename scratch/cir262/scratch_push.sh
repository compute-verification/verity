#!/usr/bin/env bash
# usage: scratch_push.sh FILE...   (writes each FILE as scratch/cir262/<basename> on the scratch branch and pushes it)
set -euo pipefail
R=/agent/repos/verity
B=cursor/circuits-cir262-scratch-6f04
export GIT_INDEX_FILE=/tmp/cir262/scratch.index
if git -C "$R" rev-parse -q --verify "refs/heads/$B" > /dev/null; then parent=$(git -C "$R" rev-parse "refs/heads/$B"); else parent=$(git -C "$R" rev-parse "origin/$B"); fi
git -C "$R" read-tree "$parent"
for f in "$@"; do
  f=$(realpath "$f")
  mode=100644; [ -x "$f" ] && mode=100755
  blob=$(git -C "$R" hash-object -w "$f")
  git -C "$R" update-index --add --cacheinfo "$mode,$blob,scratch/cir262/$(basename "$f")"
done
tree=$(git -C "$R" write-tree)
if [ "$tree" = "$(git -C "$R" rev-parse "$parent^{tree}")" ]; then echo "nothing new at $parent"; exit 0; fi
commit=$(git -C "$R" commit-tree "$tree" -p "$parent" -m "cir262 scratch: $# file(s)")
git -C "$R" update-ref "refs/heads/$B" "$commit"
git -C "$R" push -q -u origin "$B" 2>&1 | tail -2
echo "pushed $commit"
