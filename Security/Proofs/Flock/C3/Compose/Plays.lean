import Proofs.Flock.C3.Compose.Honest
import Proofs.Flock.Soundness.Refine.LiveSim

/-!
# A stage's decode is its play

A game plays on at most one prefix of a tape, to one output (`playsT_prefix`), so `stageOf` at a tape a game plays a
prefix of is that play (`stageOf_of_playsT`). The read facts' dischargers (`hSparse_of_read`, `hRegLevels_of_read`)
decode their stages through it.
-/

namespace FlockC3.Compose

open FlockSoundness FlockSoundness.Game

/-- **A game plays on at most one prefix of a tape**, to one output. -/
theorem playsT_prefix {α : Type} {g : Game α} {p : List Step} {a : α} (ha : PlaysT g p a) :
    ∀ {p' : List Step} {a' : α} {l : List Step}, PlaysT g p' a' → p <+: l → p' <+: l → p' = p ∧ a' = a := by
  induction ha with
  | ret a => exact fun h _ _ => h.ret_inv
  | send m _ ih =>
      intro p' a' l h hp hp'
      cases p' with
      | nil => cases h
      | cons s p' =>
          obtain ⟨m', rfl, hk⟩ := h.send_inv
          obtain ⟨t, rfl⟩ := hp
          obtain ⟨e, hq⟩ := List.cons_prefix_cons.mp hp'
          obtain rfl : m' = m := eq_of_heq (Sigma.mk.inj (Sum.inl.inj e)).2
          obtain ⟨rfl, rfl⟩ := ih hk (List.prefix_append _ t) hq
          exact ⟨rfl, rfl⟩
  | coin x _ ih =>
      intro p' a' l h hp hp'
      cases p' with
      | nil => cases h
      | cons s p' =>
          obtain ⟨x', rfl, hk⟩ := h.coin_inv
          obtain ⟨t, rfl⟩ := hp
          obtain ⟨e, hq⟩ := List.cons_prefix_cons.mp hp'
          obtain rfl : x' = x := eq_of_heq (Sigma.mk.inj (Sum.inr.inj e)).2
          obtain ⟨rfl, rfl⟩ := ih hk (List.prefix_append _ t) hq
          exact ⟨rfl, rfl⟩

/-- **A stage's decode is its play** on any prefix of the tape it plays on. -/
theorem stageOf_of_playsT {α : Type} {g : Game α} {l p : List Step} {a : α} (d : α) (h : PlaysT g p a)
    (hp : p <+: l) : stageOf g l d = (p, a) := by
  have hex : ∃ q : List Step × α, q.1 <+: l ∧ PlaysT g q.1 q.2 := ⟨(p, a), hp, h⟩
  unfold stageOf
  rw [dite_eq_left hex]
  obtain ⟨e₁, e₂⟩ := playsT_prefix h hex.choose_spec.2 hp hex.choose_spec.1
  exact Prod.ext e₁ e₂

end FlockC3.Compose
