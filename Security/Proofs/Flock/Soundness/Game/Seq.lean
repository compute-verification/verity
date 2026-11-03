import Proofs.Core.Game.Seq
import Definitions.Flock.Game.Seq
import Proofs.Flock.Soundness.Game.Runs

/-!
# C-Flock's game library is Core's: `Seq`

Every name here is an alias (`export`) of Core's declaration in `Proofs.Core.Game.Seq`, so C-Flock's soundness proofs
and every other protocol's read one game library.
-/

namespace FlockSoundness.Game

export Verity.Game (expect_bind expect_draw expect_map expect_runsG)

end FlockSoundness.Game
