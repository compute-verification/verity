module

public import Definitions.Core.Game
public import Definitions.Core.Game.Batch
public import Definitions.Core.Game.Expect
public import Definitions.Core.Game.Lock
public import Definitions.Core.Game.Prob
public import Definitions.Core.Game.Seq
public import Definitions.Flock.Rewinding
public import Mathlib.Algebra.BigOperators.Group.Finset.Defs
public import Mathlib.Basic.ENNReal.Basic
public import Mathlib.Data.Fintype.Card

/-!
# Flock's game vocabulary is Core's

Every name here is an alias (`export`) of Core's definition in `Definitions.Core.Game` and its `Batch`, `Expect`,
`Lock`, `Prob` and `Seq`, so Flock's spec and every other protocol's read one game library. Flock's own game
vocabulary, the bind tape and `seqG`, is `Definitions.Flock.Game.Tape`.
-/

@[expose] public section

namespace FlockSoundness

export Verity (Game)

end FlockSoundness

namespace FlockSoundness.Game

export Verity.Game (avg batch bind casesOn coin draw drawStrat expect expect₂ ind instMonad interleave interleaveWith
  Lock map prCoin prob rec recOn recv ret runsG runsStrat send Strategy value)

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

namespace FlockSoundness.Game.Strategy

export Verity.Game.Strategy (bind map ofBind ofMap)

end FlockSoundness.Game.Strategy

namespace FlockSoundness.Game.Lock

export Verity.Game.Lock (Aligned batchLock bind casesOn Chooser coin Disjoint dropDummy embedLeft embedRight forkE
  getPair getPair_fresh₁ getPair_fresh₂ getPair_keep₁ getPair_keep₂ getPair_set₁ getPair_set₂ interleaveLock leafE
  leftSame lockWith lockWithL lockWithSame map mapL mono oracleStrat rec recOn ret rightSame RPair same sameThen send
  table withDummy)

end FlockSoundness.Game.Lock

namespace FlockSoundness.Game.Lock.Aligned

export Verity.Game.Lock.Aligned (casesOn coin rec recOn ret send table)

end FlockSoundness.Game.Lock.Aligned

namespace FlockSoundness.Game.Lock.Disjoint

export Verity.Game.Lock.Disjoint (casesOn left mk rec recOn right)

end FlockSoundness.Game.Lock.Disjoint

namespace FlockSoundness.Rewinding

export Verity.Rewinding (IsPluralityE)

end FlockSoundness.Rewinding
