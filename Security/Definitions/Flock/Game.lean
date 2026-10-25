import Definitions.Core.Game

/-!
# C-Flock's game vocabulary is Core's: `Game`

Every name here is an alias (`export`) of Core's definition in `Definitions.Core.Game`, so C-Flock's spec and every
other protocol's read one game library.
-/

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
