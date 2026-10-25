import Proofs.Core.Game.Rewinding

/-!
# C-Flock's rewinding lemma is Core's

Every name here is an alias (`export`) of Core's declaration in `Proofs.Core.Game.Rewinding`, so C-Flock's soundness
proofs and every other protocol's read one game library.
-/

namespace FlockSoundness.Rewinding

export Verity.Rewinding (ConflictAt IsPlurality Off bad bad_sq_le conflict exists_plurality off_eq off_le_row
  off_sq_le opened openedTo openedTo_subset prBad_le sum_conflictAt_le)

end FlockSoundness.Rewinding
