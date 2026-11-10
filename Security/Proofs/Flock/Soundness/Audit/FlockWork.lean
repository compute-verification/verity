module

public import Flock.Draw
public import Proofs.Flock.Soundness.Audit.Work
import Mathlib.Algebra.Order.Algebra
import Mathlib.Analysis.SpecialFunctions.Pow.NNReal

/-!
# The work sampling design's rule is the executable's

`Flock.Draw.workK` sizes the executable's work samples (`flock-verify draw --work`) above each stratum's floor from the
verifier's work table, and U2 holds each stratum of a work sample to it. It is `Law.workRule`, the rule the theorems of
`Audit/Work.lean` read; `Flock.Draw.stratumK`, the count law's rule, is `Law.countRule`.
-/

@[expose] public section

namespace FlockSoundness.Audit.Law

theorem workRule_eq_draw : Flock.Draw.workK = workRule := rfl

theorem countRule_eq_draw : Flock.Draw.stratumK = countRule := rfl

end FlockSoundness.Audit.Law
