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

theorem ofFn_update {A : Type*} {J : ℕ} (f : Fin J → A) (j : Fin J) (a : A) :
    List.ofFn (Function.update f j a) = (List.ofFn f).take j ++ a :: (List.ofFn f).drop (j + 1) := by
  apply List.ext_getElem
  · simp only [List.length_ofFn, List.length_append, List.length_take, List.length_cons, List.length_drop]
    omega
  · intro i h₁ h₂
    simp only [List.length_ofFn] at h₁
    rw [List.getElem_ofFn]
    rcases lt_trichotomy i j with hlt | rfl | hgt
    · rw [List.getElem_append_left (by simp only [List.length_take, List.length_ofFn]; omega), List.getElem_take, List.getElem_ofFn,
        Function.update_of_ne (Fin.ne_of_lt (by exact hlt))]
    · rw [List.getElem_append_right (by simp only [List.length_take, List.length_ofFn]; omega)]
      simp
    · rw [List.getElem_append_right (by simp only [List.length_take, List.length_ofFn]; omega), Function.update_of_ne (Fin.ne_of_gt (by exact hgt))]
      simp only [List.length_take, List.length_ofFn, List.getElem_cons, List.getElem_drop, List.getElem_ofFn]
      split
      · omega
      congr 1
      ext
      simp only
      omega

end FlockSoundness.Discharge.Tables
