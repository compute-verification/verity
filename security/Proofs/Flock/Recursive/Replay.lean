module

public import Proofs.Flock.Soundness.Refine.LiveSim

/-! Replaying a run on one tape (`Game.playT`), so a finder reads an outcome off a transcript without choosing one. -/

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

theorem playT_eq_some {α : Type} {g : Game α} {ts : List Step} {a : α} : playT g ts = some a ↔ PlaysT g ts a :=
  ⟨playsT_of_playT g ts a, playT_of_playsT⟩

end FlockSoundness.Game
