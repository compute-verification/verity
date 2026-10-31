import Definitions.Core.Game.Lock
import Definitions.Flock.Game.Expect
import Definitions.Flock.Game.Interleave

/-!
# C-Flock's game vocabulary is Core's: `Lock`

Every name here is an alias (`export`) of Core's definition in `Definitions.Core.Game.Lock`, so C-Flock's spec and every
other protocol's read one game library.
-/

namespace FlockSoundness.Game

export Verity.Game (Lock)

end FlockSoundness.Game

namespace FlockSoundness.Game.Lock

export Verity.Game.Lock (Aligned bind casesOn Chooser coin Disjoint dropDummy embedLeft embedRight forkE getPair
  getPair_fresh₁ getPair_fresh₂ getPair_keep₁ getPair_keep₂ getPair_set₁ getPair_set₂ interleaveLock leafE lockWith map
  mono oracleStrat rec recOn ret RPair same send table withDummy)

end FlockSoundness.Game.Lock

namespace FlockSoundness.Game.Lock.Aligned

export Verity.Game.Lock.Aligned (casesOn coin rec recOn ret send table)

end FlockSoundness.Game.Lock.Aligned

namespace FlockSoundness.Game.Lock.Disjoint

export Verity.Game.Lock.Disjoint (casesOn left mk rec recOn right)

end FlockSoundness.Game.Lock.Disjoint
