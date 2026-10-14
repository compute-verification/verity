#!/usr/bin/env bash
set -euo pipefail
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
export TMPDIR="$d"
rc=0
python3 tools/verity/lean/lean_changed.py --records Security --update --out "$OUT/security" 2> >(tee "$d/records.log" >&2) || rc=$?
tree=$(grep -o 'audit.py exit [0-9]* in [0-9]* s on [^ ]*' "$d/records.log" | head -1 | awk '{print $NF}' || true)
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
if [ -n "$tree" ] && [ -d "$tree/Security/Proofs" ]; then
  (cd "$tree/Security/Proofs" && lake env lean "$d/Probe.lean") > "$OUT/probes.txt" 2>&1 || echo "probe exit $?" >> "$OUT/probes.txt"
else
  echo "no records tree found in the log (got '$tree')" > "$OUT/probes.txt"
fi
exit $rc
