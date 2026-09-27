/- `lake env lean CheckAxioms.lean`: the axioms the L3-F theorems depend on (the test allows only Lean's standard three). -/
import FlockLevel3
#print axioms FlockLevel3.clmul_spec
#print axioms FlockLevel3.reduce_spec
#print axioms FlockLevel3.F128.poly_add
#print axioms FlockLevel3.F128.poly_mul
#print axioms FlockLevel3.F128.poly_injective
#print axioms FlockLevel3.toAdj_injective
#print axioms FlockLevel3.toAdj_surjective
#print axioms FlockLevel3.G_irreducible
#print axioms FlockLevel3.GF128.add_eq
#print axioms FlockLevel3.GF128.mul_eq
#print axioms FlockLevel3.GF128.inv_eq
#print axioms FlockLevel3.GF128.xinv_mul_x
#print axioms FlockLevel3.phi_injective
#print axioms FlockLevel3.phi_surjective
#print axioms FlockLevel3.GF256.add_eq
#print axioms FlockLevel3.GF256.mul_eq
#print axioms FlockLevel3.T_xK
#print axioms FlockLevel3.Q_irreducible
