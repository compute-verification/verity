#!/usr/bin/env bash
set -euo pipefail
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
export TMPDIR="$d"
export PATH="$HOME/.elan/bin:$PATH"
rc=0
python3 tools/verity/lean/lean_changed.py --records Security --update --out "$OUT/security" || rc=$?
cat > "$d/Probe.lean" <<'EOF'
import Proofs.Flock.C3.Compose.Discharge
import Proofs.Flock.C3.Compose.RowsReg
#print axioms FlockC3.PubRows.checkLabels_at
#print axioms FlockC3.PubRows.drawJ_tableDraw
#print axioms FlockC3.PubRows.label_of_checked
#print axioms FlockC3.PubRows.hPubRows_of_check'
#print axioms FlockC3.PubRows.hPubRows_of_checked
#print axioms FlockC3.PubRows.checksAccepted_checked
#print axioms FlockC3.Compose.hbr_honest_reg
EOF
files="Security/Proofs/Flock/Recursive/AuditC3.lean Security/Proofs/Flock/C3/PubRows/Check.lean Security/Proofs/Flock/C3/PubRows/Honest.lean Security/Proofs/Flock/C3/PubRows/Shape.lean verity/core/service/flock/Flock/RecVStar.lean"
tree=""
for t in "$HOME"/.cache/verity-check/lean-records/build-*; do
  ok=1
  for f in $files; do cmp -s "$f" "$t/$f" || ok=0; done
  if [ "$ok" = 1 ]; then tree="$t"; break; fi
done
if [ -n "$tree" ]; then
  echo "tree $tree" > "$OUT/probes.txt"
  (cd "$tree/Security/Proofs" && lake env lean "$d/Probe.lean") >> "$OUT/probes.txt" 2>&1 || echo "probe exit $?" >> "$OUT/probes.txt"
else
  echo "no records tree holds this commit's files" > "$OUT/probes.txt"
fi
(cd verity/core/service/flock && lake build flock-verify Flock.Test.Labels) > "$OUT/flock-build.txt" 2>&1 || { echo "flock build exit $?" >> "$OUT/flock-build.txt"; rc=1; }
pick() {
  local want
  want=$(python3 -c "import json;print(json.load(open('verity/ml/flock/verifier/upstream.json'))['$1']['sha256'])")
  for f in "$RESEARCH_RUN_DIR"/inputs/*; do
    [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$want" ] && { echo "$f"; return 0; }
  done
  return 0
}
up=$(pick upstream); in=$(pick inputs)
if [ -n "$up" ] && [ -n "$in" ] && [ -x verity/core/service/flock/.lake/build/bin/flock-verify ]; then
  mkdir -p "$d/agree"
  python3 tools/verity/check/check.py --lean-agreement "$d/agree" "$up" "$in" > "$OUT/agreement.txt" 2>&1 \
    && echo "agreement passed" >> "$OUT/agreement.txt" || { echo "agreement exit $?" >> "$OUT/agreement.txt"; rc=1; }
  find "$d/agree/flock-agreement" -maxdepth 1 -type f -name '*.json' -exec cp {} "$OUT/" \;
else
  echo "agreement not run: upstream=$up inputs=$in" > "$OUT/agreement.txt"; rc=1
fi
exit $rc
