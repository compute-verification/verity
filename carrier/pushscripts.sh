#!/bin/bash
# pushscripts.sh MESSAGE: commit the files of /tmp/carrier/scripts as carrier/ on top of the tip of
# cursor/proofs-carrier-scripts-9257 on origin, under the checkout's own identity, and print the commit. The caller pushes
# it: git push origin <commit>:refs/heads/cursor/proofs-carrier-scripts-9257 (a fast-forward, never forced).
set -euo pipefail
R=/agent/repos/verity; B=cursor/proofs-carrier-scripts-9257
tip=$(git -C $R ls-remote origin refs/heads/$B | cut -f1)
git -C $R cat-file -e "$tip^{commit}" 2>/dev/null || git -C $R fetch -q origin $B
idx=$(mktemp -u /tmp/carrier/index.XXXXXX)
GIT_INDEX_FILE=$idx git -C $R read-tree "$tip"
for f in /tmp/carrier/scripts/*; do
  [ -f "$f" ] || continue
  h=$(git -C $R hash-object -w "$f")
  GIT_INDEX_FILE=$idx git -C $R update-index --add --cacheinfo "100644,$h,carrier/$(basename "$f")"
done
tree=$(GIT_INDEX_FILE=$idx git -C $R write-tree)
rm -f "$idx"
if [ "$tree" = "$(git -C $R rev-parse "$tip^{tree}")" ]; then echo "nothing new: $tip"; exit 0; fi
c=$(git -C $R commit-tree "$tree" -p "$tip" -m "$1")
git -C $R log -1 --format='%H %an <%ae> | %cn <%ce>' "$c"
