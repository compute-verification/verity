#!/usr/bin/env bash
set -euo pipefail

python3 tools/verity/lean/lean_changed.py --records Security --update --out "$OUT/security"

cat > /tmp/k2-probes.lean <<'EOF'
import Proofs.Flock.C3.Sparse.SessionZero
import Proofs.Flock.Recursive.HPortTie
import Proofs.Flock.Recursive.AuditC3Proof

#print axioms FlockSoundness.Discharge.Recursion.record_decode_of_load
#print axioms FlockC3.Sparse.record_decodes_of_accepted
#print axioms FlockC3.session_h_of_sessionAt
#print axioms FlockC3.PointsFromHistory
#print axioms Flock.SecurityProofs.recursiveAuditC3At
EOF

(cd Security/Proofs && lake env lean /tmp/k2-probes.lean) | tee "$OUT/probes.txt"
