module

public import Definitions.Flock.Audit.Harm
public import Proofs.Flock.Soundness.Audit.Law

/-!
# Harm bounds

A consumer weights an incorrect proof block by what it costs it: its output bits for exfiltration, its work for compute. `IsHarmBound
L h δ H` is the specification of core's `verity.core.protocols.profile.Stratified.harm_bound`, "an upper bound on
`sum_{u in B} harm(u)` over the incorrect sets `B` with `escape >= delta`". The Python function computes one by a greedy
fill in floating point; this file states what it must return and does not prove that implementation. `Audit/HarmBound.lean`
proves it for the stratified sampling design at closed-form bounds: the greedy fill's value, over the risk `k_s c / n_s` or the exact
risk, at the price where the fill stops.
-/

@[expose] public section

namespace FlockSoundness.Audit

open Finset
open scoped ENNReal
variable {n : ℕ}

end FlockSoundness.Audit
