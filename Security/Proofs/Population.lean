import Proofs.Population.Determinacy
import Proofs.Population.Necessity
import Proofs.Population.Egress
import Proofs.Population.Pointer
import Proofs.Population.Leakage
import Proofs.Population.Work
import Proofs.Population.Instance
import Proofs.Population.LongValue

/-!
# Populations: which object each policy needs of sampled proofs' guarantee

`Determinacy`: robust determinacy and the downstream cone from single assignment and acyclicity, and the honest
evaluation. `Necessity`: determinacy for every choice of unit functions holds exactly when the checker's two checks do.
`Egress`: the egress is a function of the wrong units' writes when the secrets lie outside its upstream cone, so it takes
at most `(∑_{j ≤ m} C(n, j)) · |V| ^ (m w)` values; the location term and acyclicity are each needed. `Pointer`: a
time-ordered machine trace keeps the count, but with value-dependent reads one wrong one-bit unit makes the egress
copy `N` secret words its honest run never reads; declared statically, the checker sees them upstream of the egress.
`Leakage`: beyond the honest egress, the egress takes at most `C(|U|, m) · |V| ^ (m E_max)` values, `E(u)` the egress
slots in `u`'s downstream cone, read off the wiring; the bound is attained, and a machine's step has `E_max = |E|`. In
bounded-reach terms, control reach is at most `min(m w, Reach_inf) + log C(|U|, ≤ m)` and leakage about the secrets at
most `Reach_inf + log C(|U|, ≤ m)`, `Reach_inf` the egress slots inside the wrong units' cone.
Equivalently, control times amplification: at most `K · α` values, `α = 1` when no secret is upstream of the egress, and
fixed arithmetic at fixed indices amplifies one wrong bit into `N` secret bits.
`Work`: every skipped unit is downstream of a wrong unit or lucky, so the skipped work is at most `m F` plus the lucky
work, `F` bounding a unit's downstream cone, or `F · wt(B)` plus the lucky work with each cone at most `F` times its
root's weight; a copied unit is correct work never done, so luck needs distinct units.
`Instance`: a circuit's partition is a single-assignment population whose wrong units are the partition's, and an
acyclic one when its units are ordered along the wires (intervals in layout order, as `Q_call` v1 cuts); a unit that
sandwiches another on a path of wires, as `Q_word` v1 can cut, makes a cycle.
`LongValue`: a committed value of another length than its slot's width buys nothing. The verifier refuses such an
opening before it reads it, so the slot is ⊥ in its view (or the run holds a collision), and its writer and readers are
wrong; ⊥ is weakly dominated by a short wrong value, and the theorems above hold over `Option V` with `|V| + 1` for `|V|`.
-/
