import Definitions.Flock.Game.Basic

/-! The definitions of `Proofs.Flock.Soundness.Game.Interleave` that C-Flock's guarantees read, or that a
definition here uses. The lemmas about them stay there. -/

namespace FlockSoundness

open scoped ENNReal

namespace Game

variable {α β : Type}

export Verity.Game (interleaveWith interleave)

end Game

end FlockSoundness
