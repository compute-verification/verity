module

public import Definitions.Flock.Audit.Harm
public import Proofs.Flock.Soundness.Audit.SamplingDesign

/-! Harm bounds: `IsHarmBound` specifies what core's `Stratified.harm_bound` returns, and this file does not prove that implementation. -/

@[expose] public section

namespace FlockSoundness.Audit

open Finset
open scoped ENNReal
variable {n : ℕ}

end FlockSoundness.Audit
