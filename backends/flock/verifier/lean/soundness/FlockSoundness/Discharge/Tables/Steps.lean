import FlockSoundness.Refine.LiveSim
import FlockSoundness.Game.Batch
import FlockSoundness.Protocol.Discharge.Tables.Steps

/-!
# The model's moves inside a batch

A session of `J` tables runs `batch` over the tables, each table the `interleave` of its two reps. The simulation of a
live session moves one rep of one table at a time, so it needs the model's moves at a position deep inside that
structure as moves of the whole game.

* `Steps g g'`: a path of the model's messages from `g` to `g'` (data: the messages), and `Sim.ofSteps` plays it.
* `Ctx C`: a context `C` in which a message at the hole is a path of the whole game's messages, and a coin at the hole
  is, after such a path, a coin of the whole game. Contexts compose (`Ctx.comp`), and `bind`, either side of
  `interleave`, and a position of `batch` are contexts.
* `steps_batch_ofFn`: a batch whose every position has a path to an end has a path to the end with their outputs.
-/

namespace FlockSoundness.Discharge.Tables

open Game Refine

variable {α β γ : Type}

namespace Steps

end Steps

namespace Ctx

variable {A B : Type}

/-- The hole. -/
def id : Ctx (fun g : Game A => g) where
  send _ m := .send m (.refl _)
  coin k := ⟨k, .refl _, fun _ => rfl⟩

end Ctx

end FlockSoundness.Discharge.Tables
