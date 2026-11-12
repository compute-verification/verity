#!/usr/bin/env bash
# usage: checkpoint.sh "LINE"   (appends LINE to scratch/cir262/checkpoint.md on the scratch branch and pushes it)
set -euo pipefail
R=/agent/repos/verity
B=cursor/circuits-cir262-scratch-6f04
F=scratch/cir262/checkpoint.md
line=$1
export GIT_INDEX_FILE=/tmp/cir262/scratch.index
if git -C "$R" rev-parse -q --verify "refs/heads/$B" > /dev/null; then parent=$(git -C "$R" rev-parse "refs/heads/$B"); else parent=$(git -C "$R" rev-parse origin/main); fi
git -C "$R" read-tree "$parent"
old=$(git -C "$R" show "$parent:$F" 2> /dev/null || true)
blob=$(printf '%s%s\n' "${old:+$old
}" "$line" | git -C "$R" hash-object -w --stdin)
git -C "$R" update-index --add --cacheinfo "100644,$blob,$F"
tree=$(git -C "$R" write-tree)
commit=$(git -C "$R" commit-tree "$tree" -p "$parent" -m "cir262 checkpoint")
git -C "$R" update-ref "refs/heads/$B" "$commit"
git -C "$R" push -q -u origin "$B" 2>&1 | tail -2
echo "pushed $commit"
