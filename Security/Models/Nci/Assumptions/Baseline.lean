import Definitions.Nci.Dark

/-!
# NCI's baseline rule: the named premise

What the protocol lets a declaration's baseline `θ` hold, a premise about the protocol rather than about any circuit.
Every bound through a containment, total or partial (`PartialContainment`), rests on it.
-/

namespace Nci.Assumptions

open Verity

/-- **The baseline is registered** ([BASELINE-REGISTERED]; CMP-66). The protocol's baseline rule: every entry of a
declaration's baseline `θ` holds registered public data, registered before the run's commitment, and the reference
circuit `c` takes as its own baseline `c.θ` exactly the inputs it reads of that kind. An output of an earlier run,
certified or not, is never baseline (Daniel, 21 Oct: registration happens at the start of a deployment and essentially
never changes). So no value computed off the books, before the run or during it, is ever registered as baseline, and
neither is any novel input of `c`.

What the bounds use is the rule's consequence for a partial containment `M` of `c` in `d`: a gate that `M`'s computed
gates `S` read from outside `S` (`c.D.into M.S`: a dark gate of `c`, or an input gate of `c`) goes into `d`'s baseline
only when it is in `c`'s baseline. A dark value, or a novel input of `c`, that `d` listed in `θ` would be read for free,
and an IU could then thread `c`'s counted multiplications on it without paying for the read.

The NCI check does not enforce the rule: it takes `θ` from the instance and checks only that each entry is an input gate,
with no duplicates. Commitments are hiding, so it cannot tell a registered value from one computed beforehand on
unmonitored hardware (gradients, partial sums, updated weights). Until it checks each entry's provenance, this is a
premise of every guarantee through a containment: a partial one, or a total one `M : Containment c.D d.D` as the partial
containment with no dark gates, `PartialContainment.ofContainment M`, for which it says that no novel input of `c` that
`c` reads is in `d`'s baseline. -/
def BaselineRegistered (c d : Declaration) (M : PartialContainment c d) : Prop :=
  ∀ a ∈ c.D.into M.S, M.emb a ∈ d.θ → a ∈ c.θ

end Nci.Assumptions
