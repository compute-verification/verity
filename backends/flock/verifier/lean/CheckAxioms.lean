/- `lake env lean CheckAxioms.lean`: the axioms every level-3 theorem depends on (the test allows only Lean's standard three). -/
import FlockProofs
#print axioms Flock.climb_binding
#print axioms Flock.opens_binding
#print axioms Flock.merkleCheck_opens
#print axioms Flock.merkle_binding
#print axioms Flock.squeeze_answers
#print axioms Flock.squeeze_retained
