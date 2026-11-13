#!/bin/bash
# blocks.sh OURS THEIRS [BASE]: print the conflict blocks of the merge of OURS with THEIRS (with BASE as the merge base when given)
ours=$1; theirs=$2; base=${3:-}
if [ -n "$base" ]; then out=$(git merge-tree --write-tree --name-only --no-messages --merge-base=$base $ours $theirs 2>/dev/null); else out=$(git merge-tree --write-tree --name-only --no-messages $ours $theirs 2>/dev/null); fi
t=$(echo "$out" | head -1)
echo "merged tree $t; conflicted files: $(echo "$out" | tail -n +2 | grep -c .)"
echo "$out" | tail -n +2 | while read -r f; do [ -z "$f" ] && continue; echo "=== $f: $(git show $t:"$f" | grep -c '^<<<<<<< ') block(s)"; git show $t:"$f" | awk '/^<<<<<<< /{p=1} p{print} /^>>>>>>> /{p=0; print "-----"}' | cut -c1-${COLS:-200} | head -${LINES_MAX:-80}; done
