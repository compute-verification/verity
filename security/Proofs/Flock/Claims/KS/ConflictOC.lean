module

public import Proofs.Flock.Claims.Deploy

/-! Runs of the verifier over `OC` within a log (`Runs`), peeled step by step. -/

@[expose] public section

namespace FlockDeploy.KS

open Flock
open Claims (OC OCV)

def Runs (H : Claims.Msg → Claims.Msg) {α : Type} (x : OCV α) (a : α) (L : List Claims.Msg) : Prop :=
  (x.run.eval H).1 = .ok a ∧ ∀ p ∈ (x.run.eval H).2, p.1 ∈ L

theorem Runs.self (H : Claims.Msg → Claims.Msg) {α : Type} {x : OCV α} {a : α} (h : (x.run.eval H).1 = .ok a) :
    Runs H x a ((x.run.eval H).2.map Prod.fst) :=
  ⟨h, fun _ hp => List.mem_map_of_mem hp⟩

theorem Runs.mono {H : Claims.Msg → Claims.Msg} {α : Type} {x : OCV α} {a : α} {L L' : List Claims.Msg}
    (h : Runs H x a L) (hL : ∀ m ∈ L, m ∈ L') : Runs H x a L' :=
  ⟨h.1, fun p hp => hL _ (h.2 p hp)⟩

theorem Runs.bind {H : Claims.Msg → Claims.Msg} {α β : Type} {x : OCV α} {f : α → OCV β} {b : β}
    {L : List Claims.Msg} (h : Runs H (x >>= f) b L) : ∃ a, Runs H x a L ∧ Runs H (f a) b L := by
  obtain ⟨h1, h2⟩ := h
  simp only [ExceptT.run_bind, Claims.OC.bind_eq, Claims.OC.eval_bind] at h1 h2
  rcases hx : x.run.eval H with ⟨rx, lx⟩
  rw [hx] at h1 h2
  cases rx with
  | error e => cases h1
  | ok a =>
    refine ⟨a, ⟨by rw [hx], fun p hp => h2 p (List.mem_append_left _ ?_)⟩, h1,
      fun p hp => h2 p (List.mem_append_right _ hp)⟩
    rw [hx] at hp
    exact hp

theorem Runs.sbind {H : Claims.Msg → Claims.Msg} {σ α β : Type} {x : StateT σ OCV α} {f : α → StateT σ OCV β}
    {s : σ} {b : β × σ} {L : List Claims.Msg} (h : Runs H ((x >>= f) s) b L) :
    ∃ a s₁, Runs H (x s) (a, s₁) L ∧ Runs H (f a s₁) b L := by
  obtain ⟨⟨a, s₁⟩, h1, h2⟩ := Runs.bind (x := x s) (f := fun p => f p.1 p.2) h
  exact ⟨a, s₁, h1, h2⟩

theorem Runs.lbind {H : Claims.Msg → Claims.Msg} {σ τ α β : Type} {x : StateT σ (StateT τ OCV) α}
    {f : α → StateT σ (StateT τ OCV) β} {s : σ} {t : τ} {b : (β × σ) × τ} {L : List Claims.Msg}
    (h : Runs H ((x >>= f) s t) b L) :
    ∃ a s₁ t₁, Runs H (x s t) ((a, s₁), t₁) L ∧ Runs H (f a s₁ t₁) b L := by
  obtain ⟨⟨a, s₁⟩, t₁, h1, h2⟩ := Runs.sbind (x := x s) (f := fun p => f p.1 p.2) (s := t) h
  exact ⟨a, s₁, t₁, h1, h2⟩

theorem Runs.lift {H : Claims.Msg → Claims.Msg} {α : Type} {v : Flock.V α} {a : α} {L : List Claims.Msg}
    (h : Runs H (monadLift v : OCV α) a L) : v = .ok a := h.1

theorem Runs.pure {H : Claims.Msg → Claims.Msg} {α : Type} {a a' : α} {L : List Claims.Msg}
    (h : Runs H (Pure.pure a : OCV α) a' L) : a' = a := by
  have := h.1
  cases this
  rfl

def shadow (H : Claims.Msg → Claims.Msg) : With.Sim OCV Flock.V where
  R x y := (x.run.eval H).1 = y
  pure _ := rfl
  bind {α β x y f g} hxy hfg := by
    simp only [ExceptT.run_bind, Claims.OC.bind_eq, Claims.OC.eval_bind]
    rcases hx : x.run.eval H with ⟨rx, lx⟩
    rw [hx] at hxy
    dsimp only at hxy ⊢
    subst hxy
    cases rx with
    | error e => rfl
    | ok a => exact hfg a
  throw _ := rfl
  lift _ := rfl

theorem shadow_std : (shadow sha512).Hashes Claims.Hashes.oc With.std where
  sha512 b := by
    show Except.ok (Claims.hOf sha512 b) = _
    rw [hOf_sha512]
    rfl
  sha256 _ := rfl
  blake3 _ := rfl
  blake3Keyed _ _ := rfl

theorem Runs.std {H : Claims.Msg → Claims.Msg} {α : Type} {x : OCV α} {y : Flock.V α} {a : α} {L : List Claims.Msg}
    (hR : (shadow H).R x y) (h : Runs H x a L) : y = .ok a :=
  hR.symm.trans h.1

end FlockDeploy.KS
