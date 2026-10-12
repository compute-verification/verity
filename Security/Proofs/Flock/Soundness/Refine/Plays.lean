import Proofs.Flock.Soundness.Refine.Run
import Proofs.Flock.Soundness.Game.Interleave

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

/-- **Runs compose.** A run of `g` to `a`, then a run of `f a`, is a run of `g.bind f`. -/
theorem bind {g : Game α} {f : α → Game β} {ms₁ ms₂ : List Msg} {cs₁ cs₂ : List (Array W)} {a : α} {b : β}
    (h₁ : Plays ι g ms₁ cs₁ a) (h₂ : Plays ι (f a) ms₂ cs₂ b) : Plays ι (g.bind f) (ms₁ ++ ms₂) (cs₁ ++ cs₂) b := by
  induction h₁ with
  | ret a => exact h₂
  | send m _ ih => exact .send m (ih h₂)
  | coinVec c hc h1 _ ih => exact .coinVec c hc h1 (ih h₂)
  | coinOne c hc _ ih => exact .coinOne c hc (ih h₂)

/-- `bind` with the tapes given by their concatenations. -/
theorem bind' {g : Game α} {f : α → Game β} {ms₁ ms₂ ms : List Msg} {cs₁ cs₂ cs : List (Array W)} {a : α} {b : β}
    (h₁ : Plays ι g ms₁ cs₁ a) (h₂ : Plays ι (f a) ms₂ cs₂ b) (hms : ms = ms₁ ++ ms₂) (hcs : cs = cs₁ ++ cs₂) :
    Plays ι (g >>= f) ms cs b := hms ▸ hcs ▸ bind h₁ h₂

theorem pure (a : α) : Plays ι (Pure.pure a : Game α) [] [] a := .ret a

theorem recv (M : Type) (m : M) : Plays ι (Game.recv M) [⟨M, m⟩] [] m := .send m (.ret m)

theorem drawVec {n : ℕ} [Fintype (Fin n → F)] [Nonempty (Fin n → F)] (c : Array W) (hc : c.size = n)
    (h1 : n ≠ 1) : Plays ι (Game.draw (Fin n → F)) [] [c] (fun i => ι c[i.val]!) := .coinVec c hc h1 (.ret _)

theorem drawOne [Fintype F] [Nonempty F] (c : Array W) (hc : c.size = 1) :
    Plays ι (Game.draw F) [] [c] (ι c[0]!) := .coinOne c hc (.ret _)

/-- A run of `g` is a run of `g.map f`, to the image of its output. -/
theorem map {g : Game α} {ms : List Msg} {cs : List (Array W)} {a : α} (f : α → β) (h : Plays ι g ms cs a) :
    Plays ι (g.map f) ms cs (f a) := by
  have := bind (f := fun a => Game.ret (f a)) h (Plays.ret (ι := ι) (f a))
  simpa [Game.map] using this

/-- `interleave` against a finished second interaction runs the first alone. -/
theorem interleave_ret (g₁ : Game α) (b : β) : interleave g₁ (.ret b) = g₁.map fun a => (a, b) := by
  cases g₁ <;> rfl

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
