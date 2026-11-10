module

public import Definitions.Nci.Assumptions.Info

/-!
# The inference-only policy over threads, reads counted by information: the guarantees

A set whose fresh input reads are a state that is a function of `k` bits and an ingress counted by width reads at most
`k + width(ingress)` bits of information (`InfoOfState`). Theorem 1 with reads counted by information is stated under
`WeightsSupportFmt2` (`Definitions.Nci.Guarantees.Fmt2`).
-/

@[expose] public section

namespace Nci.Guarantees

open Verity Finset

/-- **Reads bounded by a state**: if a set `R` of computed gates reads, outside the registered information, only gates
of a state `Sst` and an ingress `Sin`, and the state's values over the runs with the registered information at `θv` are
a function `f` of `k` bits, then `R`'s gates take at most `2^(k + width(Sin))` values and its fresh input reads carry at
most `k + width(Sin)` bits of information. The baseline is what is registered: after a wipe, a state that is registered
data entangled with at most `k` bits of slack reads as `k` bits, however wide it is. -/
def InfoOfState : Prop :=
  ∀ (D : RegisteredCircuit) (θv : D.D.Trace) (R Sst Sin : Finset (Fin D.D.N)) (k : ℕ)
    (f : BitVec k → (g : Sst) → BitVec (D.D.w g)),
    R ⊆ D.D.computed → D.D.into R \ D.θ ⊆ Sst ∪ Sin → (∀ a ∈ admissible D θv, ∃ s, values Sst a = f s) →
      support D θv R ≤ 2 ^ (k + D.D.width Sin) ∧ novelInfo D θv R ≤ k + D.D.width Sin

end Nci.Guarantees
