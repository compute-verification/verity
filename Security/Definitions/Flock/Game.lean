module

public import Definitions.Core.Game
public import Mathlib.Algebra.BigOperators.Group.Finset.Defs
public import Mathlib.Basic.ENNReal.Basic
public import Mathlib.Data.Fintype.Card
import Mathlib.Algebra.Order.Module.Field
import Mathlib.Basic.ENNReal.Real

/-!
# Flock's game vocabulary is Core's: `Game`

Every name here is an alias (`export`) of Core's definition in `Definitions.Core.Game`, so Flock's spec and every
other protocol's read one game library.
-/

@[expose] public section

namespace FlockSoundness

export Verity (Game)

end FlockSoundness

namespace FlockSoundness.Game

export Verity.Game (avg bind casesOn coin draw instMonad map prCoin prob rec recOn recv ret send Strategy value)

end FlockSoundness.Game

namespace FlockSoundness.Game.coin

export Verity.Game.coin (inj injEq)

end FlockSoundness.Game.coin

namespace FlockSoundness.Game.ret

export Verity.Game.ret (inj injEq)

end FlockSoundness.Game.ret

namespace FlockSoundness.Game.send

export Verity.Game.send (inj injEq)

end FlockSoundness.Game.send
