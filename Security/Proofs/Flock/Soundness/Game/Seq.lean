module

public import Proofs.Core.Game.Seq
public import Definitions.Flock.Game.Seq
public import Proofs.Flock.Soundness.Game.Runs
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Tactic.Positivity.Finset

/-!
# Flock's game library is Core's: `Seq`

Every name here is an alias (`export`) of Core's declaration in `Proofs.Core.Game.Seq`, so Flock's soundness proofs
and every other protocol's read one game library.
-/

@[expose] public section

namespace FlockSoundness.Game

export Verity.Game (expect_bind expect_draw expect_map expect_runsG)

end FlockSoundness.Game
