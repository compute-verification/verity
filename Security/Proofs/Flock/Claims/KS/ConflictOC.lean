import Proofs.Flock.Claims.Deploy

/-!
# Runs of the verifier over `OC`, peeled

`Runs H x a L`: the oracle program `x : OCV α`, against `H`, returns `a` and queries only messages of `L`. A run of a
bind is a run of each part (`Runs.bind`, and through the transcript's and Ligerito's state, `Runs.sbind`,
`Runs.lbind`), so a run of the shipped verdict within its own log is a run of each step within that log. A step's
verdict is the step's with the hashes `hOf H` (`shadow`, a `Flock.With.Sim`), which at `sha512` is the compiled
verifier's own (`Runs.std`).
-/

namespace FlockDeploy.KS

open Flock
open Claims (OC OCV)

/-- **A run of `x` against `H`**: it returns `a`, and every query it makes is a message of `L`. -/
def Runs (H : Claims.Msg → Claims.Msg) {α : Type} (x : OCV α) (a : α) (L : List Claims.Msg) : Prop :=
  (x.run.eval H).1 = .ok a ∧ ∀ p ∈ (x.run.eval H).2, p.1 ∈ L

/-- Every `OCV` program runs within its own log. -/
theorem Runs.self (H : Claims.Msg → Claims.Msg) {α : Type} {x : OCV α} {a : α} (h : (x.run.eval H).1 = .ok a) :
    Runs H x a ((x.run.eval H).2.map Prod.fst) :=
  ⟨h, fun _ hp => List.mem_map_of_mem hp⟩

theorem Runs.mono {H : Claims.Msg → Claims.Msg} {α : Type} {x : OCV α} {a : α} {L L' : List Claims.Msg}
    (h : Runs H x a L) (hL : ∀ m ∈ L, m ∈ L') : Runs H x a L' :=
  ⟨h.1, fun p hp => hL _ (h.2 p hp)⟩

/-- **A run of a bind is a run of each part.** -/
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

/-- A run of a bind through a state. -/
theorem Runs.sbind {H : Claims.Msg → Claims.Msg} {σ α β : Type} {x : StateT σ OCV α} {f : α → StateT σ OCV β}
    {s : σ} {b : β × σ} {L : List Claims.Msg} (h : Runs H ((x >>= f) s) b L) :
    ∃ a s₁, Runs H (x s) (a, s₁) L ∧ Runs H (f a s₁) b L := by
  obtain ⟨⟨a, s₁⟩, h1, h2⟩ := Runs.bind (x := x s) (f := fun p => f p.1 p.2) h
  exact ⟨a, s₁, h1, h2⟩

/-- A run of a bind through the Ligerito state and the transcript. -/
theorem Runs.lbind {H : Claims.Msg → Claims.Msg} {σ τ α β : Type} {x : StateT σ (StateT τ OCV) α}
    {f : α → StateT σ (StateT τ OCV) β} {s : σ} {t : τ} {b : (β × σ) × τ} {L : List Claims.Msg}
    (h : Runs H ((x >>= f) s t) b L) :
    ∃ a s₁ t₁, Runs H (x s t) ((a, s₁), t₁) L ∧ Runs H (f a s₁ t₁) b L := by
  obtain ⟨⟨a, s₁⟩, t₁, h1, h2⟩ := Runs.sbind (x := x s) (f := fun p => f p.1 p.2) (s := t) h
  exact ⟨a, s₁, t₁, h1, h2⟩

/-- A lifted verdict queries nothing. -/
theorem Runs.lift {H : Claims.Msg → Claims.Msg} {α : Type} {v : Flock.V α} {a : α} {L : List Claims.Msg}
    (h : Runs H (monadLift v : OCV α) a L) : v = .ok a := h.1

theorem Runs.pure {H : Claims.Msg → Claims.Msg} {α : Type} {a a' : α} {L : List Claims.Msg}
    (h : Runs H (Pure.pure a : OCV α) a' L) : a' = a := by
  have := h.1
  cases this
  rfl

/-! ## A run's verdict is the step's with the hashes `hOf H` -/

/-- **An `OCV` program and a verdict**: the program's verdict against `H` is the verdict. -/
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

theorem shadow_hashes (H : Claims.Msg → Claims.Msg) :
    (shadow H).Hashes Claims.Hashes.oc (Hashes.ofSha512 (Claims.hOf H)) where
  sha512 _ := rfl
  sha256 _ := rfl
  blake3 _ := rfl
  blake3Keyed _ _ := rfl

/-- At `sha512` the hashes are the deployed ones. -/
theorem shadow_std : (shadow sha512).Hashes Claims.Hashes.oc With.std where
  sha512 b := by
    show Except.ok (Claims.hOf sha512 b) = _
    rw [hOf_sha512]
    rfl
  sha256 _ := rfl
  blake3 _ := rfl
  blake3Keyed _ _ := rfl

/-- A run's verdict is the verdict the shadow relates it to. -/
theorem Runs.std {H : Claims.Msg → Claims.Msg} {α : Type} {x : OCV α} {y : Flock.V α} {a : α} {L : List Claims.Msg}
    (hR : (shadow H).R x y) (h : Runs H x a L) : y = .ok a :=
  hR.symm.trans h.1

/-- A step of a bind whose verdict the shadow knows. -/
theorem Runs.bind_std {H : Claims.Msg → Claims.Msg} {α β : Type} {x : OCV α} {y : Flock.V α} {f : α → OCV β} {b : β}
    {L : List Claims.Msg} (hR : (shadow H).R x y) (h : Runs H (x >>= f) b L) : ∃ a, y = .ok a ∧ Runs H (f a) b L :=
  let ⟨a, h1, h2⟩ := Runs.bind h
  ⟨a, Runs.std hR h1, h2⟩

/-- A transcript step whose verdict the shadow knows. -/
theorem Runs.sbind_std {H : Claims.Msg → Claims.Msg} {σ α β : Type} {x : StateT σ OCV α} {y : StateT σ Flock.V α}
    {f : α → StateT σ OCV β} {s : σ} {b : β × σ} {L : List Claims.Msg} (hR : ((shadow H).st σ).R x y)
    (h : Runs H ((x >>= f) s) b L) : ∃ a s₁, y s = .ok (a, s₁) ∧ Runs H (f a s₁) b L :=
  let ⟨a, s₁, h1, h2⟩ := Runs.sbind h
  ⟨a, s₁, Runs.std (hR s) h1, h2⟩

/-- A Ligerito step whose verdict the shadow knows. -/
theorem Runs.lbind_std {H : Claims.Msg → Claims.Msg} {σ τ α β : Type} {x : StateT σ (StateT τ OCV) α}
    {y : StateT σ (StateT τ Flock.V) α} {f : α → StateT σ (StateT τ OCV) β} {s : σ} {t : τ} {b : (β × σ) × τ}
    {L : List Claims.Msg} (hR : (((shadow H).st τ).st σ).R x y) (h : Runs H ((x >>= f) s t) b L) :
    ∃ a s₁ t₁, y s t = .ok ((a, s₁), t₁) ∧ Runs H (f a s₁ t₁) b L :=
  let ⟨a, s₁, t₁, h1, h2⟩ := Runs.lbind h
  ⟨a, s₁, t₁, Runs.std (hR s t) h1, h2⟩

end FlockDeploy.KS
