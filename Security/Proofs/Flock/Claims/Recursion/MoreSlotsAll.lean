import Proofs.Flock.Claims.Recursion.PastIn
import Proofs.Flock.Claims.Recursion.AccFits
import Proofs.Flock.Claims.Recursion.KSLin2LayR

/-!
# `MoreSlotsTrue` at the outer relation, unconditionally

`moreSlotsTrue` off its two leaves, both proved: `pastIn` (an output row's port is past the unit's inputs) and
`accFits` (a statement reads `acc` only below its instances).
-/

namespace RecDeploy.MoreSlots

theorem moreSlotsTrue_deployed (st : Stage) :
    RecDeploy.MoreSlotsTrue (deployZkRec _root_.Flock.Readable.relation) st :=
  moreSlotsTrue (pastIn st) accFits

end RecDeploy.MoreSlots
