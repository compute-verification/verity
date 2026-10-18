import Proofs.Population.Determinacy
import Proofs.Population.Necessity
import Proofs.Population.Egress
import Proofs.Population.Pointer
import Proofs.Population.Leakage
import Proofs.Population.Work
import Proofs.Population.Instance
import Proofs.Population.Oblivious

/-!
# Populations: which object each policy needs of sampled proofs' guarantee

`Determinacy`: robust determinacy and the downstream cone from single assignment and acyclicity, and the honest
evaluation. `Necessity`: determinacy for every choice of unit functions holds exactly when the checker's two checks do.
`Egress`: the egress is a function of the wrong units' writes when the secrets lie outside its upstream cone, so it takes
at most `(∑_{j ≤ m} C(n, j)) · |V| ^ (m w)` values; the location term and acyclicity are each needed. `Pointer`: a
time-ordered machine trace keeps the count, but with value-dependent reads one wrong one-bit unit makes the egress
copy `N` secret words its honest run never reads; declared statically, the checker sees them upstream of the egress.
`Leakage`: the egress over any set of secrets takes at most `K · α` values, the control count `K` times a bound `α` on
what the rest of the population amplifies the wrong units' writes into; `α = 1` when no secret is upstream of the
egress, and fixed arithmetic at fixed indices amplifies one wrong bit into `N` secret bits.
`Work`: every skipped unit is downstream of a wrong unit or lucky, so the skipped work is at most `m F` plus the lucky
work, `F` bounding a unit's downstream cone, or `F · wt(B)` plus the lucky work with each cone at most `F` times its
root's weight; a copied unit is correct work never done, so luck needs distinct units.
`Instance`: a circuit's partition is a single-assignment population whose wrong units are the partition's, and an
acyclic one when its units are ordered along the wires (intervals in layout order, as `Q_call` v1 cuts); a unit that
sandwiches another on a path of wires, as `Q_word` v1 can cut, makes a cycle.
`Oblivious`: every time-ordered trace pads to a checked population with its own wrong units, and an oblivious one (read
sets independent of every value, or of all but the public inputs) is that population with nothing added. Locality and
the skipped-work cone hold in the honest run's own dataflow with no obliviousness, but that cone is not public, and the
pointer machine is oblivious in the classical sense yet leaks. With the address slots honest, a run is an assignment of
the honest run's frozen dataflow.
-/
