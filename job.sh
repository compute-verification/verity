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
#print axioms FlockSoundness.Discharge.ZkReg.drawn_of_acceptsZKR
#print axioms FlockC3.Compose.labelsSelf
#print axioms FlockC3.Compose.ownReads_cover
#print axioms FlockC3.Compose.registrantCovers_attained
#print axioms FlockC3.PubRows.hPubRows_of_check'
#print axioms FlockC3.PubRows.hPubRows_of_checked
#print axioms FlockC3.Compose.hbr_honest_reg
EOF
files="Security/Proofs/Flock/C3/Compose/RowsReg.lean Security/Proofs/Flock/C3/PubRows/Check.lean Security/Proofs/Flock/C3/PubRows/Honest.lean Security/Proofs/Flock/C3/PubRows/Shape.lean"
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
exit $rc
