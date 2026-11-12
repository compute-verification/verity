#!/usr/bin/env bash
# usage: checkpoint.sh "one line"  -- commits scratch/cir258/checkpoint.md on cursor/circuits-cir258-scratch-6f04 without a checkout
set -euo pipefail
G="git -C /agent/repos/verity"
B=cursor/circuits-cir258-scratch-6f04
$G fetch -q origin "$B" 2>/dev/null && parent=$($G rev-parse FETCH_HEAD) || parent=$($G rev-parse origin/main)
printf '%s\n' "$1" > /tmp/gemmcall/checkpoint.md
blob=$($G hash-object -w /tmp/gemmcall/checkpoint.md)
export GIT_INDEX_FILE=/tmp/gemmcall/scratch.idx
$G read-tree "$parent"
$G update-index --add --cacheinfo "100644,$blob,scratch/cir258/checkpoint.md"
tree=$($G write-tree)
c=$($G commit-tree "$tree" -p "$parent" -m "GEMMCALL checkpoint: $1")
$G push -q origin "$c:refs/heads/$B"
echo "$c"
