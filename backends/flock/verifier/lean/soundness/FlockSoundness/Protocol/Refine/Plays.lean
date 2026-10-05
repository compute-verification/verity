import FlockSoundness.Protocol.Refine.Run
import FlockSoundness.Protocol.Game.Interleave

/-!
# Runs compose

`Plays.bind`: a run of `g` to `a`, then a run of `f a`, is a run of `g.bind f`; with `pure`, `recv`, `draw` and `map`
these build a run of any model program step by step. `Plays.interleave_first` runs two interactions concurrently with
the first one finished before the second one moves, which is how the executable replays a table's two reps, each on its
own stream.
-/

namespace FlockSoundness.Game

variable {W F : Type} [Inhabited W] {ι : W → F}

namespace Plays

variable {α β : Type}

/-- `interleave` against a finished second interaction runs the first alone. -/
theorem interleave_ret (g₁ : Game α) (b : β) : interleave g₁ (.ret b) = g₁.map fun a => (a, b) := by
  cases g₁ <;> rfl

end Plays

end FlockSoundness.Game
