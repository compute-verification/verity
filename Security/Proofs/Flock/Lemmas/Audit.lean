module

public import Mathlib.Analysis.MeanInequalities

/-!
# Flock's one-stage audit: lemmas

These are steps of proofs, not guarantees: they sit with the proofs, and no record of the lock pins them.

Each guarantee is a `def G : Prop`, with the docstring, binders and `open`s of the statement in its source
module. Its proof is the theorem `FlockSoundness.Audit.SecurityProofs.G : FlockSoundness.Audit.Guarantees.G`.
-/

@[expose] public section

namespace FlockSoundness.Audit.Law
open scoped ENNReal NNReal

/-- **The sizing of record** (`δ = 2^-40`, `ε = 0.1%`): `(1 − 1/1000)^27713 ≤ 2^-40`. -/
def _root_.FlockSoundness.Audit.Guarantees.Law.RecordSizing : Prop :=
  ((999 : ℝ≥0∞) / 1000) ^ 27713 ≤ ((2 : ℝ≥0∞) ^ 40)⁻¹

end FlockSoundness.Audit.Law

namespace FlockSoundness.Audit.Law
open scoped ENNReal NNReal

/-- One sample fewer does not meet it under the work bound: `(1 − 1/1000)^27712 > 2^-40`. -/
def _root_.FlockSoundness.Audit.Guarantees.Law.RecordSizingOneFewer : Prop :=
  ¬ ((999 : ℝ≥0∞) / 1000) ^ 27712 ≤ ((2 : ℝ≥0∞) ^ 40)⁻¹

end FlockSoundness.Audit.Law
