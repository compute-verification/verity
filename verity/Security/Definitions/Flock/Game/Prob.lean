import Definitions.Flock.Game.Interleave

/-! The definitions of `Proofs.Flock.Soundness.Game.Prob` that C-Flock's guarantees read, or that a
definition here uses. The lemmas about them stay there. -/

namespace FlockSoundness.Game

open scoped ENNReal

variable {α β : Type}

export Verity.Game (Strategy.ofMap)

end FlockSoundness.Game
