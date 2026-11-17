module

public import Mathlib.Analysis.MeanInequalities

/-! Steps of the proofs of Flock's one-stage audit, not guarantees, so no record of the lock pins them. -/

@[expose] public section

namespace FlockSoundness.Audit.SamplingDesign
open scoped ENNReal NNReal

def _root_.FlockSoundness.Audit.Guarantees.SamplingDesign.RecordSizing : Prop :=
  ((999 : ℝ≥0∞) / 1000) ^ 27713 ≤ ((2 : ℝ≥0∞) ^ 40)⁻¹

end FlockSoundness.Audit.SamplingDesign

namespace FlockSoundness.Audit.SamplingDesign
open scoped ENNReal NNReal

def _root_.FlockSoundness.Audit.Guarantees.SamplingDesign.RecordSizingOneFewer : Prop :=
  ¬ ((999 : ℝ≥0∞) / 1000) ^ 27712 ≤ ((2 : ℝ≥0∞) ^ 40)⁻¹

end FlockSoundness.Audit.SamplingDesign
