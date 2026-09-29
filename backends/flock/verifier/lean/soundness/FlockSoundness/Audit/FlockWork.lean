import Flock.Draw
import FlockSoundness.Audit.Work

/-!
# The work law's rule is the executable's

`Flock.Draw.workK` sizes the executable's work draws (`flock-verify draw --work`), and U2 holds each stratum of a work draw
to it. It is `Law.workRule`, the rule that `Law.workK` applies and the theorems of `Audit/Work.lean` read.
-/

namespace FlockSoundness.Audit.Law

theorem workRule_eq_draw : Flock.Draw.workK = workRule := rfl

end FlockSoundness.Audit.Law
