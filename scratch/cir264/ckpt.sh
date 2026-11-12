#!/usr/bin/env bash
# Appends one line to scratch/cir264/checkpoint.md on the scratch branch and pushes it, without touching the checkout; each
# FILE named after the text goes into the same commit as scratch/cir264/<its name>:
#   bash ckpt.sh "TEXT" [FILE...]
# The line's time is read here (date -u): never type one in TEXT.  The commit's parent is the branch's head on origin, so
# the files it does not name stay.  Never --force.
set -euo pipefail
R=${REPO:-/agent/repos/verity}
B=cursor/circuits-cir264-scratch-6f04
P=scratch/cir264
text=$1; shift
line="$(date -u +%Y-%m-%dT%H:%MZ) $text"
git -C "$R" fetch -q origin "$B" 2>/dev/null && parent=$(git -C "$R" rev-parse FETCH_HEAD) || parent=$(git -C "$R" rev-parse origin/main)
old=$(git -C "$R" show "$parent:$P/checkpoint.md" 2>/dev/null || true)
tmp=$(mktemp)
{ [ -n "$old" ] && printf '%s\n' "$old"; printf '%s\n' "$line"; } > "$tmp"
idx=$(mktemp -u)
GIT_INDEX_FILE=$idx git -C "$R" read-tree "$parent"
blob=$(git -C "$R" hash-object -w "$tmp")
GIT_INDEX_FILE=$idx git -C "$R" update-index --add --cacheinfo "100644,$blob,$P/checkpoint.md"
for f in "$@"; do
  [ -f "$f" ] || { echo "ckpt: no file $f"; rm -f "$idx" "$tmp"; exit 2; }
  blob=$(git -C "$R" hash-object -w "$(realpath "$f")")
  GIT_INDEX_FILE=$idx git -C "$R" update-index --add --cacheinfo "100644,$blob,$P/$(basename "$f")"
done
tree=$(GIT_INDEX_FILE=$idx git -C "$R" write-tree)
commit=$(git -C "$R" commit-tree "$tree" -p "$parent" -m "CIR-264 scratch: $text")
rm -f "$idx" "$tmp"
git -C "$R" push -q origin "$commit:refs/heads/$B"
echo "pushed $commit to $B (parent $parent): $line"
