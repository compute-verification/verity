#!/usr/bin/env bash
set -euo pipefail

if [[ "${K2_TARGET:-}" == 1 ]]; then
  cd Security/Proofs
  lake build Proofs.Flock.Recursive.ValueInner Proofs.Flock.C3.Sparse.SessionZero \
    Proofs.Flock.Recursive.HPortTie Proofs.Flock.Recursive.AuditC3Proof
  exit
fi

if [[ "${K2_PROBES_ONLY:-}" != 1 ]]; then
  python3 tools/verity/lean/lean_changed.py --records verity/core/service/flock --update --out "$OUT/flock"
  python3 tools/verity/lean/lean_changed.py --records Security --update --out "$OUT/security"
  python3 tools/verity/lean/lean_changed.py --records Security/Properties --update --out "$OUT/properties"
fi

cat > /tmp/k2-probes.lean <<'EOF'
import Proofs.Flock.C3.Sparse.SessionZero
import Proofs.Flock.Recursive.HPortTie
import Proofs.Flock.Recursive.AuditC3Proof

#print axioms FlockSoundness.Discharge.Recursion.record_decode_of_load
#print axioms FlockC3.Sparse.record_decodes_of_accepted
#print axioms FlockC3.HPort.session_h_of_sessionAt
#print axioms FlockC3.HPort.PointsFromHistory
#print axioms Flock.SecurityProofs.recursiveAuditC3At
EOF

(
  if [[ -n "${K2_SEED_LAKE:-}" ]]; then
    mkdir -p Security/Proofs/.lake
    cp -a --reflink=auto "$K2_SEED_LAKE/." Security/Proofs/.lake/
  fi
  cd Security/Proofs
  lake build Proofs.Flock.Recursive.ValueInner Proofs.Flock.C3.Sparse.SessionZero \
    Proofs.Flock.Recursive.HPortTie Proofs.Flock.Recursive.AuditC3Proof
  lake env lean /tmp/k2-probes.lean
) | tee "$OUT/probes.txt"
