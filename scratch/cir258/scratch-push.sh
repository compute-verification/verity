#!/usr/bin/env bash
# usage: scratch-push.sh "message" FILE...  -- commits each FILE as scratch/cir258/<basename> on cursor/circuits-cir258-scratch-6f04
set -euo pipefail
G="git -C /agent/repos/verity"
B=cursor/circuits-cir258-scratch-6f04
msg=$1; shift
$G fetch -q origin "$B" 2>/dev/null && parent=$($G rev-parse FETCH_HEAD) || parent=$($G rev-parse origin/main)
export GIT_INDEX_FILE=/tmp/gemmcall/scratch.idx
$G read-tree "$parent"
for f in "$@"; do
  blob=$($G hash-object -w "$f")
  $G update-index --add --cacheinfo "100644,$blob,scratch/cir258/$(basename "$f")"
done
tree=$($G write-tree)
c=$($G commit-tree "$tree" -p "$parent" -m "GEMMCALL scratch: $msg")
$G push -q origin "$c:refs/heads/$B"
echo "$c"
