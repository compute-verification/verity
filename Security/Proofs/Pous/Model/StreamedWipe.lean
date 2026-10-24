import Definitions.Pous.Model.StreamedWipe

/-! The proof-only part of `Definitions.Pous.Model.StreamedWipe`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pous.StreamedWipe

open scoped ENNReal

/-- The ideal game: the probability that `P` is accepted and keeps `z`, with uniform labels. -/
noncomputable def winIdeal {N k b q M E : ℕ} (P : Prover N k b q M E) : ℝ≥0∞ :=
  pr (wipeDistinguisher P)

end Pous.StreamedWipe
