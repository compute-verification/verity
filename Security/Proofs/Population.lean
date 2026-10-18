import Proofs.Population.Determinacy
import Proofs.Population.Necessity
import Proofs.Population.Egress
import Proofs.Population.Pointer
import Proofs.Population.Work

/-!
# Populations: which object each policy needs of sampled proofs' guarantee

`Determinacy`: robust determinacy and the downstream cone from single assignment and acyclicity, and the honest
evaluation. `Necessity`: determinacy for every choice of unit functions holds exactly when the checker's two checks do.
`Egress`: the egress is a function of the wrong units' writes when the secrets lie outside its upstream cone, so it takes
at most `(∑_{j ≤ m} C(n, j)) · |V| ^ (m w)` values; the location term and acyclicity are each needed. `Pointer`: a
time-ordered machine trace keeps the count, but with value-dependent reads one wrong one-bit unit makes the egress
copy `N` secret words its honest run never reads; declared statically, the checker sees them upstream of the egress.
`Work`: every skipped unit is downstream of a wrong unit or lucky, so the skipped work is at most `m F` plus the lucky
work, `F` bounding a unit's downstream cone; a copied unit is correct work never done, so luck needs distinct units.
-/
