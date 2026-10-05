import FlockSoundness.Refine.Run
import FlockSoundness.Game.Interleave
import FlockSoundness.Protocol.Refine.Plays

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

/-- **The first interaction first.** When the second interaction has not finished, `interleave` can run the first one to
its end, with the prover choosing it at every step, and then the second one. The coins are the first's, then the
second's. -/
theorem interleave_first {g₁ : Game α} {g₂ : Game β} {ms₁ ms₂ : List Msg} {cs₁ cs₂ : List (Array W)} {a : α} {b : β}
    (h₁ : Plays ι g₁ ms₁ cs₁ a) (h₂ : Plays ι g₂ ms₂ cs₂ b) :
    ∃ ms, Plays ι (interleave g₁ g₂) ms (cs₁ ++ cs₂) (a, b) := by
  cases g₂ with
  | ret b' =>
      cases h₂
      rw [interleave_ret]
      exact ⟨ms₁, by simpa using h₁.map fun a => (a, b)⟩
  | send M k₂ =>
      induction h₁ with
      | ret a => exact ⟨ms₂, h₂.map fun b => (a, b)⟩
      | send m _ ih =>
          obtain ⟨ms, h⟩ := ih
          exact ⟨⟨Bool, true⟩ :: ⟨_, m⟩ :: ms, .send true (.send m h)⟩
      | coinVec c hc h1 _ ih =>
          obtain ⟨ms, h⟩ := ih
          exact ⟨⟨Bool, true⟩ :: ms, .send true (.coinVec c hc h1 h)⟩
      | coinOne c hc _ ih =>
          obtain ⟨ms, h⟩ := ih
          exact ⟨⟨Bool, true⟩ :: ms, .send true (.coinOne c hc h)⟩
  | coin C k₂ =>
      induction h₁ with
      | ret a => exact ⟨ms₂, h₂.map fun b => (a, b)⟩
      | send m _ ih =>
          obtain ⟨ms, h⟩ := ih
          exact ⟨⟨Bool, true⟩ :: ⟨_, m⟩ :: ms, .send true (.send m h)⟩
      | coinVec c hc h1 _ ih =>
          obtain ⟨ms, h⟩ := ih
          exact ⟨⟨Bool, true⟩ :: ms, .send true (.coinVec c hc h1 h)⟩
      | coinOne c hc _ ih =>
          obtain ⟨ms, h⟩ := ih
          exact ⟨⟨Bool, true⟩ :: ms, .send true (.coinOne c hc h)⟩

end Plays

end FlockSoundness.Game
