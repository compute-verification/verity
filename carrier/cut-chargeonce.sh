#!/bin/bash
# cut-chargeonce.sh [--dry]: once main holds ChargeOnce.lean as #9513's head has it, cut
# cursor/proofs-carrier-chargeonce-terms-9257 from main in a worktree of its own (/tmp/carrier/wt-chargeonce) with the
# scratch commit's change as one commit under the checkout's own identity, and print its head and main's commit. It
# refuses, changing nothing, when main has no such file, when main's file or its entry is not what the change was made
# on, when the list changes by more than the entry's line, or when the lint's count of the file's sites is not the entry it writes.
# It pushes nothing: the caller pushes the branch.
set -euo pipefail
R=/agent/repos/verity; B=${CUT_BRANCH:-cursor/proofs-carrier-chargeonce-terms-9257}; WT=${CUT_WT:-/tmp/carrier/wt-chargeonce}
OLD=a2643447fa06a78b6467633fbbf6ebfd2f1c6d42; NEW=a05073714cbf2b4d8e25555a0975e646189a0a79
F=security/Proofs/Flock/Claims/ChargeOnce.lean; L=tests/lint/allowlists/repo_retired_terms.json
cd $R
git fetch -q origin main
M=${CUT_MAIN:-$(git rev-parse origin/main)}   # CUT_MAIN, CUT_BRANCH, CUT_WT: a rehearsal on a local merge, never pushed
git cat-file -e "$OLD^{commit}" 2>/dev/null && git cat-file -e "$NEW^{commit}" 2>/dev/null || git fetch -q origin cursor/proofs-carrier-9513-terms-scratch-9257
if git merge-base --is-ancestor $OLD "$M"; then echo "main $M holds #9513's head $OLD"; else echo "main $M does not hold #9513's head $OLD"; fi
git cat-file -e "$M:$F" 2>/dev/null || { echo "REFUSED: main has no $F: #9513 is not on it"; exit 3; }
[ "$(git rev-parse "$M:$F")" = "$(git rev-parse "$OLD:$F")" ] || { echo "REFUSED: main's $F is not the file the change was made on; make the change again by hand"; exit 4; }
git show "$M:$L" | grep -qF "{\"file\": \"$F\", \"count\": 5}" || { echo "REFUSED: main's entry for $F is not 5"; exit 4; }
[ "${1:-}" = "--dry" ] && { echo "dry: would cut $B from $M"; exit 0; }
[ -e "$WT" ] && { echo "REFUSED: $WT exists"; exit 5; }
git worktree add -q -b $B "$WT" "$M"
cd "$WT"
# The file is the scratch commit's whole (main's is the one it was made on). The list's entry is rewritten as one line,
# not patched: main's list has other lines beside it than #9513's had.
git show "$NEW:$F" > $F
python3 - "$L" "$F" <<'PY'
import sys
path, f = sys.argv[1:]
old, new = f'{{"file": "{f}", "count": 5}}', f'{{"file": "{f}", "count": 2}}'
text = open(path).read()
assert text.count(old) == 1, f"{path}: {text.count(old)} lines hold the entry of 5"
open(path, "w").write(text.replace(old, new))
PY
git add $F $L
[ "$(git rev-parse :$F)" = "$(git rev-parse "$NEW:$F")" ] || { echo "REFUSED: the file is not the scratch commit's"; exit 6; }
[ "$(git diff --cached --numstat "$M" -- $L | cut -f1,2)" = "1	1" ] || { echo "REFUSED: the list changes by more than its one line"; exit 6; }
git commit -q -F - <<'EOF'
ChargeOnce.lean: its three theorems in a namespace of their own, and its retired-terms entry goes from 5 to 2

The three theorems were declared in the audit theorems' namespace, from a module outside that directory. They are now
in FlockSoundness.Audit.ChargeOnce, and the module opens the audit theorems' namespace for the two definitions its one
statement reads, Adapted and Predictable. The module docstring names the audit theorems' module by what it holds. No
statement's text changes, no other file names the three theorems, and no lock records them.

Two sites of the retired word stay, both names main declares and other files read: the import of the audit theorems'
module, and the line that opens its namespace.
EOF
H=$(git rev-parse HEAD)
SITES_ROOT=$WT python3 /tmp/carrier/scripts/sites.py $H $F | head -4
n=$(SITES_ROOT=$WT python3 /tmp/carrier/scripts/sites.py $H $F | head -1 | sed 's/.*: \([0-9]*\) site.*/\1/')
[ "$n" = 2 ] || { echo "REFUSED: the lint counts $n sites, not 2"; exit 7; }
git log -1 --format='%H %an <%ae> | %cn <%ce>'
git diff --stat "$M" HEAD | tail -3
echo "BASE $M"
echo "HEAD $(git rev-parse HEAD)"
