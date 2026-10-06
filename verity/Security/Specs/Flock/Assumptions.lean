import Specs.Flock.Assumptions.Soundness

/-!
# C-Flock's named assumptions (trusted)

Each named assumption is a `Prop` over `Definitions.Flock`, taken as a hypothesis by the guarantees that rest on it,
never an `axiom`. One module per topic under `Specs.Flock.Assumptions` (the soundness proof's are `Soundness`, the
recursive verifier's `Recursive`), each imported here. They import `Definitions.Flock` and each other, never a guarantee
or a proof.
-/
