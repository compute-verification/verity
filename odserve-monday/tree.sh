#!/usr/bin/env bash
# The run's tree, TREE: a detached worktree of MAIN_REPO at origin's head of REF, fetched now and clean, which must be
# WANT (the head the kit was staged for) unless WANT is set to another; then what Monday's settings and record read at it:
#   CASE-ONLY  certifier_piece.py takes --case-only (no at the kit's head: settings-monday.env runs the period in full)
#   V2-CASE    v2_case.py reads the drawn member and its layout from a5's pre-draw pin (layout_of)
#   PIN        a5's audit writes window/public-pin.json, which the onsite record holds the public file and θ's layout to
#   INODES     served_verdicts.sh refuses a serve under MIN_FREE_INODES
#   [WANT=SHA] bash tree.sh
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
ok=
for _ in 1 2 3 4; do git -C "$MAIN_REPO" fetch -q origin "$REF" && { ok=1; break; }; sleep 8; done
[ -n "$ok" ] || { echo "REFUSED: couldn't fetch $REF"; exit 3; }
head=$(git -C "$MAIN_REPO" rev-parse FETCH_HEAD)
[ -z "$WANT" ] || [ "$head" = "$WANT" ] \
  || { echo "REFUSED: origin's $REF is at $head, not the kit's $WANT: ask the coordinator, then WANT=$head bash tree.sh"; exit 3; }
if [ -d "$TREE" ]; then
  git -C "$TREE" diff --quiet HEAD || { echo "REFUSED: $TREE is dirty"; exit 3; }
  git -C "$TREE" checkout -q --detach "$head"
else
  git -C "$MAIN_REPO" worktree add -q --detach "$TREE" "$head"
fi
cd "$TREE"
yn() { if eval "$1"; then echo yes; else echo no; fi; }
echo "TREE $TREE at $(git rev-parse --short=11 HEAD) (= origin's $REF), clean: $(yn 'git diff --quiet HEAD')"
echo "CASE-ONLY $(yn "grep -q -- '\"--case-only\"' benchmarks/network_traces/certifier_piece.py")"
echo "V2-CASE drawn member, layout from the pin: $(yn "grep -q 'members=layout_of(window)' benchmarks/one_stage/v2_case.py")"
echo "PIN public-pin.json, layout held: $(yn "grep -rqs 'public-pin.json' benchmarks/one_stage/served_verdicts.sh && grep -q '^def layout_pinned' benchmarks/one_stage/onsite_record.py")"
echo "INODES MIN_FREE_INODES: $(yn "grep -q 'MIN_FREE_INODES=' benchmarks/one_stage/served_verdicts.sh")"
git log --oneline -3 | cut -c1-120
