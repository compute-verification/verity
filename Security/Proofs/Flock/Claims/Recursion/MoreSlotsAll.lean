import Proofs.Flock.Claims.Recursion.PastIn
import Proofs.Flock.Claims.Recursion.AccFits

/-!
# `MoreSlotsTrue` at the outer relation, unconditionally

`moreSlotsTrue` off its two leaves, both proved: `pastIn` (an output row's port is past the unit's inputs) and
`accFits` (a statement reads `acc` only below its instances). The statement is KSLin2LayR's `MoreSlotsTrue` at
`deployZkRec`, unfolded, so this file doesn't import KSLin2LayR.
-/

namespace RecDeploy.MoreSlots

open Flock
open Claims (Msg Out)
open Claims.Recursion.VStarB (readForms)

theorem moreSlotsTrue_deployed (st : Stage) :
    ∀ (x : Msg) (ms : List (Msg × Out)) (s : ℕ) (w : Claims.Flock.Wit) (r : XR),
      StagedPart st s x (staged st x ms s) →
      Claims.Flock.check (deployZkRec Flock.Readable.relation).parse (staged st x ms s) w = true →
        readX x = some r → ∀ p ∈ DeployedR.moreCalls r s,
          FlockDeploy.sha512 (p.1 ++ readForms w p.2.1) = readForms w p.2.2 :=
  moreSlotsTrue (pastIn st) accFits

end RecDeploy.MoreSlots
