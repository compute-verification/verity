#!/usr/bin/env bash
# Independent build and axiom audit of the Lean packages (coordinator job, PRs #85 and #89 merged with main).
#   bash tools/lean/audit.sh PKG_DIR[:CHECK_FILE] ...
# For each package: `setup.sh`, `lake build`, then `lake env lean CHECK_FILE` (default: the package's CheckAxioms.lean). Passes
# only when every build succeeds, no `.lean` file declares an `axiom`, and every `#print axioms` line in the check files reports
# nothing beyond propext, Classical.choice and Quot.sound (no sorryAx). Hypotheses such as BCHKS25 never show in `#print axioms`.
set -uo pipefail
root=$(pwd)
out=${RESEARCH_RUN_DIR:-/tmp}/lean-audit
mkdir -p "$out"
export PATH="$HOME/.elan/bin:$PATH"
fail=0
for spec in "$@"; do
  pkg=${spec%%:*}; chk=${spec#*:}; [ "$chk" = "$spec" ] && chk=CheckAxioms.lean
  tag=$(echo "$pkg" | tr '/' '_')
  echo "== $pkg ($chk)"
  bash "$root/tools/lean/setup.sh" "$root/$pkg" > "$out/$tag.setup.log" 2>&1 || { echo "AUDIT $pkg: setup failed"; fail=1; continue; }
  cd "$root/$pkg"
  s=$(date +%s)
  lake build > "$out/$tag.build.log" 2>&1; rc=$?
  echo "AUDIT $pkg: lake build rc=$rc in $(( $(date +%s) - s )) s"
  [ "$rc" = 0 ] || { tail -30 "$out/$tag.build.log"; fail=1; cd "$root"; continue; }
  if grep -rln --include='*.lean' --exclude-dir=.lake -E '^\s*axiom\s' . ; then echo "AUDIT $pkg: FAIL, an axiom is declared"; fail=1; fi
  want=$(grep -c '^#print axioms' "$chk")
  lake env lean "$chk" > "$out/$tag.axioms.txt" 2>&1; arc=$?
  [ "$arc" = 0 ] || { echo "AUDIT $pkg: FAIL, $chk did not elaborate"; tail -20 "$out/$tag.axioms.txt"; fail=1; cd "$root"; continue; }
  got=$(grep -c -E "depends on axioms|does not depend on any axioms" "$out/$tag.axioms.txt")
  [ "$got" = "$want" ] || { echo "AUDIT $pkg: FAIL, $want #print axioms lines, $got answers"; fail=1; }
  if grep -q sorryAx "$out/$tag.axioms.txt"; then echo "AUDIT $pkg: FAIL, sorryAx"; fail=1; fi
  extra=$(grep -o "depends on axioms: \[[^]]*\]" "$out/$tag.axioms.txt" | sed 's/.*\[//; s/\]//' | tr ',' '\n' | tr -d ' ' \
          | grep -v -x -E 'propext|Classical.choice|Quot.sound' | sort -u)
  [ -z "$extra" ] || { echo "AUDIT $pkg: FAIL, extra axioms: $extra"; fail=1; }
  echo "AUDIT $pkg: $want theorems audited"
  cd "$root"
done
[ "$fail" = 0 ] && echo "AUDIT: PASS" || echo "AUDIT: FAIL"
exit $fail
