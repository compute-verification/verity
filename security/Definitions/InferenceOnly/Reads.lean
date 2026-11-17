module

public import Definitions.InferenceOnly

/-! The inference-only policy in reads form: what a set of gates reads, baseline included, within the gates `G`. -/

@[expose] public section

namespace InferenceOnly

open Verity

/-- **The reads of `R` in `G`**: the width of the gates of `G` outside `R` that `R` reads, baseline included. -/
def readsIn (d : RegisteredCircuit) (R G : Finset (Fin d.D.N)) : ℕ := d.D.width (d.D.into R ∩ G)

end InferenceOnly
