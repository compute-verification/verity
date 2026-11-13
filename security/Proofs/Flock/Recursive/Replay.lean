module

public import Proofs.Flock.Soundness.Refine.LiveSim

/-!
# Recursion: a run on one tape, replayed

`Game.PlaysT g ts a` says that `g`, played with the typed steps `ts`, ends with `a`. `Game.playT g ts` computes that
outcome by replaying the steps: at a node that receives a message it takes the next step when that step is a message
of the node's type, at a node that flips a coin when it is a coin of the node's type, and it continues with the step's
value, cast along the equality of the two types. It returns nothing where a step does not fit, or where the steps and
the game do not end together.

A finder that reads the outcome of a transcript reads it with `playT`: the outcome is a function of the transcript,
and no outcome is chosen. The one test the replay decides is that a step's type is the node's. It is a proposition
about two types, decided classically, as the finders' other tests are (`StrictCR.collOf`).

* `playT_eq_some`: `playT g ts = some a` exactly when `PlaysT g ts a`.
-/

@[expose] public section

namespace FlockSoundness.Game

open Classical in
/-- **The replay of a run on one tape**: the outcome `g` ends with when it is played with the steps `ts` in their
order, and nothing when they do not play it to an outcome. -/
noncomputable def playT {α : Type} : Game α → List Step → Option α
  | .ret a, [] => some a
  | .ret _, _ :: _ => none
  | .send _ _, [] => none
  | .send M k, .inl ⟨M', m⟩ :: ts => if h : M' = M then playT (k (cast h m)) ts else none
  | .send _ _, .inr _ :: _ => none
  | @Game.coin _ _ _ _ _, [] => none
  | @Game.coin _ _ _ _ _, .inl _ :: _ => none
  | @Game.coin _ C _ _ k, .inr ⟨C', x⟩ :: ts => if h : C' = C then playT (k (cast h x)) ts else none

/-- A run on one tape is what the replay returns. -/
theorem playT_of_playsT {α : Type} {g : Game α} {ts : List Step} {a : α} (h : PlaysT g ts a) :
    playT g ts = some a := by
  induction h with
  | ret a => rw [playT]
  | send m _ ih =>
    rw [playT, dif_pos rfl]
    exact ih
  | coin x _ ih =>
    rw [playT, dif_pos rfl]
    exact ih

/-- What the replay returns is a run on the tape. -/
theorem playsT_of_playT {α : Type} (g : Game α) :
    ∀ (ts : List Step) (a : α), playT g ts = some a → PlaysT g ts a := by
  induction g with
  | ret b =>
    intro ts a h
    cases ts with
    | nil =>
      rw [playT] at h
      cases h
      exact PlaysT.ret _
    | cons s ts =>
      rw [playT] at h
      cases h
  | send M k ih =>
    intro ts a h
    cases ts with
    | nil =>
      rw [playT] at h
      cases h
    | cons s ts =>
      cases s with
      | inl mm =>
        obtain ⟨M', m⟩ := mm
        rw [playT] at h
        by_cases e : M' = M
        · rw [dif_pos e] at h
          subst e
          exact PlaysT.send m (ih _ ts a h)
        · rw [dif_neg e] at h
          cases h
      | inr _ =>
        rw [playT] at h
        cases h
  | coin C k ih =>
    intro ts a h
    cases ts with
    | nil =>
      rw [playT] at h
      cases h
    | cons s ts =>
      cases s with
      | inl _ =>
        rw [playT] at h
        cases h
      | inr xx =>
        obtain ⟨C', x⟩ := xx
        rw [playT] at h
        by_cases e : C' = C
        · rw [dif_pos e] at h
          subst e
          exact PlaysT.coin x (ih _ ts a h)
        · rw [dif_neg e] at h
          cases h

/-- **The replay returns an outcome exactly when the steps play the game to it.** -/
theorem playT_eq_some {α : Type} {g : Game α} {ts : List Step} {a : α} : playT g ts = some a ↔ PlaysT g ts a :=
  ⟨playsT_of_playT g ts a, playT_of_playsT⟩

end FlockSoundness.Game
