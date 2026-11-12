module

public import Definitions.Nci

/-!
# NCI in reads form: reads in a set of gates

`readsIn d R G` counts what `R` reads physically, baseline included, but only in the gates `G`: the width of
`in(R) ∩ G`. At `G = univ` it is at least `novelReads d R`, which leaves the baseline out.
-/

@[expose] public section

namespace Nci

open Verity

/-- **The reads of `R` in `G`**: the width of the gates of `G` outside `R` that `R` reads, baseline included. -/
def readsIn (d : RegisteredCircuit) (R G : Finset (Fin d.D.N)) : ℕ := d.D.width (d.D.into R ∩ G)

end Nci
