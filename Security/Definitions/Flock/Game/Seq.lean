module

public import Definitions.Flock.Game
public import Definitions.Flock.Game.Expect
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Tactic.Positivity.Finset
public import Definitions.Core.Game.Seq

/-!
# C-Flock's game vocabulary is Core's: `Seq`

Every name here is an alias (`export`) of Core's definition in `Definitions.Core.Game.Seq`, so C-Flock's spec and every
other protocol's read one game library.
-/

@[expose] public section

namespace FlockSoundness.Game

export Verity.Game (drawStrat runsG runsStrat)

end FlockSoundness.Game

namespace FlockSoundness.Game.Strategy

export Verity.Game.Strategy (bind map)

end FlockSoundness.Game.Strategy
