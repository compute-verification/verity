#!/usr/bin/env bash
# entry.sh FILE: append FILE to the end of inbox/proofs-coordinator.md on cursor/verity-proofs-84b5 as one commit touching
# only that file; on a rejected push, fetch and make the one commit again on the new tip (never force).
set -euo pipefail
cd /agent/repos/verity
BR=cursor/verity-proofs-84b5; F=inbox/proofs-coordinator.md; NEW="$1"; MSG="${2:-$(head -1 "$NEW" | cut -c1-200)}"
for i in 1 2 3 4 5 6; do
  git fetch -q origin "$BR"
  TIP=$(git rev-parse FETCH_HEAD)
  tmp=$(mktemp -d)
  git cat-file blob "$TIP:$F" > "$tmp/f"
  # one blank line between entries
  [ -z "$(tail -c1 "$tmp/f")" ] || echo >> "$tmp/f"
  { echo; cat "$NEW"; } >> "$tmp/f"
  blob=$(git hash-object -w "$tmp/f")
  export GIT_INDEX_FILE="$tmp/index"
  git read-tree "$TIP"
  git update-index --cacheinfo "100644,$blob,$F"
  tree=$(git write-tree)
  unset GIT_INDEX_FILE
  c=$(git commit-tree "$tree" -p "$TIP" -m "$MSG")
  n=$(git diff --name-only "$TIP" "$c" | wc -l)
  [ "$n" = 1 ] || { echo "REFUSED: the commit touches $n files"; exit 2; }
  rm -rf "$tmp"
  if git push -q origin "$c:refs/heads/$BR" 2>/tmp/carrier/push.err; then echo "pushed $c on $TIP (try $i)"; exit 0; fi
  echo "push rejected (try $i): $(tail -1 /tmp/carrier/push.err)"; sleep $((i*2))
done
exit 1
