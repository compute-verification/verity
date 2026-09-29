/- `lake env lean CheckAxioms.lean`: the axioms every level-3 theorem depends on (the test allows only Lean's standard three). -/
import FlockProofs
#print axioms Flock.Sha256.hash_size
#print axioms Flock.Sha512.hash_size
#print axioms Flock.Hm96.leafPair_collides
#print axioms Flock.Hm96.leaf_binding_of
#print axioms Flock.Hm96.Default512.leaf_binding
#print axioms Flock.rowBytes_inj
#print axioms Flock.MerkleScheme.all_sized
#print axioms Flock.climb_binding
#print axioms Flock.opens_binding
#print axioms Flock.merkleCheck_opens
#print axioms Flock.merkleCheck_sibs
#print axioms Flock.merkle_binding
#print axioms Flock.merkle_binding_collision
#print axioms Flock.MerklePair.sha256_inputs
#print axioms Flock.MerklePair.sha512_inputs
#print axioms Flock.MerklePair.hm96_inputs
#print axioms Flock.MerkleScheme.collision_hash
#print axioms Flock.squeeze_answers
#print axioms Flock.squeeze_retained
