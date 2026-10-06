import Definitions.Flock.Audit.Closure

/-! The definitions of `Proofs.Flock.Soundness.Audit.Window` that C-Flock's guarantees read, or that a
definition here uses. The lemmas about them stay there. -/

namespace FlockSoundness.Audit

open Game Finset
open scoped ENNReal NNReal

namespace Law

variable {n m C : ℕ}

/-- **Sizes that cover the weights `w` at budget `K`**: every stratum draws at least its share `K · w_s · n_s / W` of
the weight, or all its units. -/
def Covers (σ : Fin n → Fin m) (k w : Fin m → ℕ) (K : ℕ) : Prop :=
  ∀ s, K * w s * (stratum σ s).card ≤ k s * totalWork σ w ∨ k s = (stratum σ s).card

/-- Call `c`'s work `W_c`, its strata's. -/
def callWork (σ : Fin n → Fin m) (call : Fin m → Fin C) (w : Fin m → ℕ) (c : Fin C) : ℕ :=
  ∑ s ∈ univ.filter (call · = c), w s * (stratum σ s).card

/-- Call `c`'s units `N_c`. -/
def callUnits (σ : Fin n → Fin m) (call : Fin m → Fin C) (c : Fin C) : ℕ :=
  ∑ s ∈ univ.filter (call · = c), (stratum σ s).card

/-- Call `c`'s budget, its share of the window's `K`: `K_c = min(N_c, max(1, ⌈K · W_c / W⌉))`. -/
def callK (σ : Fin n → Fin m) (call : Fin m → Fin C) (w : Fin m → ℕ) (K : ℕ) (c : Fin C) : ℕ :=
  min (callUnits σ call c) (max 1 ((K * callWork σ call w c + totalWork σ w - 1) / totalWork σ w))

/-- The window's sizes: each call's work law at its own budget, `k_s = min(n_s, max(f_s, ⌈K_c · w_s · n_s / W_c⌉))`. -/
def windowK (σ : Fin n → Fin m) (call : Fin m → Fin C) (w f : Fin m → ℕ) (K : ℕ) (s : Fin m) : ℕ :=
  workRule (callK σ call w K (call s)) (w s) (stratum σ s).card (callWork σ call w (call s)) (f s)

end Law

section Audits

variable {n m : ℕ} {σ : Fin n → Fin m} {k : Fin m → ℕ} {hk : ∀ s, k s ≤ (Law.stratum σ s).card}
  {cl : Fin n → Finset (Fin n)} {Reg : Type} {session : Finset (Fin n) → Reg → Game Bool}

end Audits

end FlockSoundness.Audit
