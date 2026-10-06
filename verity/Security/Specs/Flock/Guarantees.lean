import Specs.Flock.Guarantees.Partitioning

/-!
# C-Flock's guarantees (trusted)

Every statement a document outside the package cites as proved, as a `def G : Prop` over `Definitions.Flock` and
`Specs.Flock.Assumptions`, by topic: one module per topic under `Specs.Flock.Guarantees` (the run end to end is
`EndToEnd`, the recursive verifier's is `Recursive`), each imported here. Its proof is the theorem
`Flock.SecurityProofs.G : Flock.Guarantees.G` in `Proofs.Flock` (`Proofs.Flock.EndToEnd`, `Proofs.Flock.Recursive`); a
pin cited only inside the package stays a lemma under its own name.
-/
