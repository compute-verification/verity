import Proofs.Population.Determinacy
import Proofs.Population.Necessity
import Proofs.Population.Egress
import Proofs.Population.Pointer
import Proofs.Population.Leakage
import Proofs.Population.Work
import Proofs.Population.Instance
import Proofs.Population.Oblivious
import Proofs.Population.Transfer
import Proofs.Population.Converse
import Proofs.Population.HonestCone
import Proofs.Population.LongValue
import Proofs.Population.NecessitySkeleton
import Proofs.Population.NecessityMatrix
import Proofs.Population.Exfiltration
import Proofs.Population.ExfiltrationBound
import Proofs.Population.ExfiltrationExamples
import Proofs.Population.DrawPressure
import Proofs.Population.JointPressure
import Proofs.Population.ChiPool

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
`Oblivious`: every time-ordered trace pads to a checked population with its own wrong units, and an oblivious one (read
sets independent of every value, or of all but the public inputs) is that population with nothing added. Locality and
the skipped-work cone hold in the honest run's own dataflow with no obliviousness, and the pointer machine is
oblivious in the classical sense yet leaks secrets its honest wiring never has upstream. With the address slots
honest, a run is an assignment of the honest run's frozen dataflow.
`Transfer`: every theorem above holds for a time-ordered program with the cones and upstreams of its padding, which
are public when the padded assignments are those holding the public values; that cone holds every honest run's.
`Converse`: the padding is the least wiring that expresses every program with a given address map, and the pointer
machine's honest wiring, which never has a secret upstream of the egress, does not bound its egress.
`HonestCone`: leakage in its uniform form (`card_leak_le_coneOver`) and the skipped work hold for every time-ordered,
address-closed program at the cones over its honest runs, with no obliviousness; what needs the padding is isolation,
a secret's place upstream.
`LongValue`: a committed value of another length than its slot's width buys nothing. The verifier refuses such an
opening before it reads it, so the slot is ⊥ in its view (or the run holds a collision), and its writer and readers are
wrong; ⊥ is weakly dominated by a short wrong value, and the determinacy, egress, leakage and skipped-work
theorems hold over `Option V` with `|V| + 1` for `|V|`.
`NecessitySkeleton`: tier 1 (a transcript committed before the coins and a draw fixed before it, local checks, robust
determinacy) gives the sampling certificate, and an accepted run's damaged egress lies in its wrong units' cone with
probability at least `1 − (1 − ε)^k`; a policy is inferable from the certificate exactly when its amplification is
bounded (`inference_iff`), and each tier-1 item has a counterexample family.
`NecessityMatrix`: the policy × property matrix's cells a theorem settles: integrity off the always-proved set,
control at the public widths and what a width bound buys the count draw, the late cone and commit-before-the-work-coin
each necessary for skippable work, and weight isolation by range restriction.
`Exfiltration`: under detection pressure `κ` and a per-window leak `ℓ`, what leaves through the first detection reaches
`x` with probability at most `exp (-κ (x - ℓ))` for every adaptive strategy and horizon (`exp (-κ x)` when a caught
window's egress is withheld), and at most `j (ℓ + 1/κ)` leaves on average before the `j`-th detection.
`ExfiltrationBound`: detection pressure from the independent draw and the cone, a checked population's egress changed
on at most its wrong units' cone egress, and the end-to-end bound: any decoder of the egress and a side channel `Y`
gets the target with probability at most `exp (-κ (K - Δ - ℓ)) + Σ_{j ≤ K} C(m, j) · |V| ^ K · |Y| · g`, also over a
deployment's windows.
`ExfiltrationExamples`: the stopping-time bound is attained, and without detection pressure, a halt, bound inputs or a
trusted model the leak is unbounded.
`DrawPressure`: the deployed draw, `k` of the `n` units without replacement, misses a `K`-unit wrong set with
probability `C(n - K, k) / C(n, k) ≤ (1 - k/n)^K ≤ exp (-k K / n)`, so it gives detection pressure `k/n` for a leak
of at most one per wrong unit, and with probability at least `1 - δ` what leaves through the first detection is below
`ℓ + (n/k) ln (1/δ)`.
`JointPressure`: one draw for several claims. The max rule `max_i κ_i harm_i(u)` is the least inclusion that gives
every claim its detection pressure, the union bound holds every claim's bound through the first detection at once
except with probability `Σ_i δ_i` (a claim whose count waits for the verdict without the per-window term), and a
uniform `k`-subset of `N` units escapes `b` wrong ones with probability at most `(1 - k/N)^b`.
`ChiPool`: the cheating pool with the verdict lag. With the halt `Δ` windows after the first caught window's
commitment, what leaves through the halt is at most `cum + Δ ℓ`, so under the deployed draw it is below
`(1 + Δ) ℓ + (n/k) ln (1/δ)` with probability at least `1 - δ`, and at most `j (ℓ + n/k) + Δ ℓ` on average through the
halt the `j`-th detection triggers.
-/
