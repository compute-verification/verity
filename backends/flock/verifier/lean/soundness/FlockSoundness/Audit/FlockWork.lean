import Flock.Draw
import FlockSoundness.Audit.Work

/-!
# The work law's rule is the executable's

`Flock.Draw.workK` sizes the executable's work draws (`flock-verify draw --work`) above the floor `Flock.Draw.workFloor`
gives each stratum from the law's count budget, with `Flock.Draw.stratumK` for the strata that do no work; U2 holds each
stratum of a work draw to them. They are `Law.workRule`, `Law.floorRule` and `Law.countRule`, the rules the theorems of
`Audit/Work.lean` read.
-/

namespace FlockSoundness.Audit.Law

theorem workRule_eq_draw : Flock.Draw.workK = workRule := rfl

theorem floorRule_eq_draw : Flock.Draw.workFloor = floorRule := rfl

theorem countRule_eq_draw : Flock.Draw.stratumK = countRule := rfl

end FlockSoundness.Audit.Law
