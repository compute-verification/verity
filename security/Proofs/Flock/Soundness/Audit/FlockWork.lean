module

public import Flock.Draw
public import Proofs.Flock.Soundness.Audit.Work
import Mathlib.Algebra.Order.Algebra
import Mathlib.Analysis.SpecialFunctions.Pow.NNReal

/-! The work sampling design's rule is the executable's: `Flock.Draw.workK` is `SamplingDesign.workRule`. -/

@[expose] public section

namespace FlockSoundness.Audit.SamplingDesign

theorem workRule_eq_sample : Flock.Draw.workK = workRule := rfl

theorem countRule_eq_sample : Flock.Draw.stratumK = countRule := rfl

end FlockSoundness.Audit.SamplingDesign
