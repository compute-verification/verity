module

public import Definitions.Nci.Info
public import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# The inference-only policy over threads, reads counted by information: through a containment

`RegisteredThrough` reads a bound on the reference circuit's runs at its registered values as one on a declared circuit's
runs, and `WithinBlock` bounds a set of a declared circuit's gates by an inference block's size. The guarantees over
threads with reads counted by information through a containment (`Definitions.Nci.Guarantees.Fmt2`) take both.
-/

@[expose] public section

namespace Nci.Assumptions

open Verity

/-- **`r`'s registered information is `d`'s, at the values `θv` gives it**: through the containment `M`, an input gate of
`r` whose image `d` registers is registered in `r`, and a registered gate of `r` whose image `d` registers holds there
the value `θc` commits it to. It is how a bound on `r`'s runs at `θc` reads on `d`'s runs at `θv`. -/
def RegisteredThrough {r d : RegisteredCircuit} (M : Containment r.D d.D) (θc : r.D.Trace) (θv : d.D.Trace) :
    Prop :=
  (∀ g, r.D.isInput g = true → M.emb g ∈ d.θ → g ∈ r.θ) ∧
    (∀ g ∈ r.θ, M.emb g ∈ d.θ → (θv (M.emb g)).cast (M.emb_w g) = θc g)

/-- **At most `c` computed gates**: `W` is a set of `d`'s computed gates of an inference block's size, whatever it
reads. -/
def WithinBlock (d : RegisteredCircuit) (c : ℕ) (W : Finset (Fin d.D.N)) : Prop :=
  W ⊆ d.D.computed ∧ d.D.size W ≤ c

end Nci.Assumptions
