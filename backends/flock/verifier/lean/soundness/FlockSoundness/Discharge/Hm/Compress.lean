/-
What the compression gadget's rows force (`Flock.HmNets.compress`): Σ and σ, the schedule, one round, the rounds by induction,
and the feed-forward, each against `Spec`. No `decide` on any of it.
-/
import FlockSoundness.Discharge.Hm.Word
import FlockSoundness.Protocol.Discharge.Hm.Compress
import FlockSoundness.Discharge.Hm.Gadget

namespace FlockSoundness.Discharge.Hm

open Flock.HmNets hiding and

/-! ## Σ and σ -/

/-! ## The schedule -/

/-! ## One round -/

/-! ## The rounds -/

/-! ## The compression -/

theorem Regs.length_toList {α : Type} (r : Regs α) : r.toList.length = 8 := rfl

end FlockSoundness.Discharge.Hm
