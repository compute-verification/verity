import Definitions.Flock.Audit.Stratified
import Mathlib.Analysis.MeanInequalities

/-! The definitions of `Proofs.Flock.Soundness.Audit.Work` that C-Flock's guarantees read, or that a
definition here uses. The lemmas about them stay there. -/

namespace FlockSoundness.Audit

open Game Finset
open scoped ENNReal NNReal

namespace Law

variable {n m : ℕ}

/-- The work rule, the executable's `Flock.Draw.workK`: `⌈K · work · size / total⌉`, at least `floor` and at most `size`. -/
def workRule (K work size total floor : ℕ) : ℕ := min size (max floor ((K * work * size + total - 1) / total))

/-- The work of a set of units, each doing its stratum's `w (σ u)`. -/
def workOf (σ : Fin n → Fin m) (w : Fin m → ℕ) (B : Finset (Fin n)) : ℕ := ∑ u ∈ B, w (σ u)

/-- The population's work, `W`. -/
abbrev totalWork (σ : Fin n → Fin m) (w : Fin m → ℕ) : ℕ := workOf σ w univ

/-- Stratum `s`'s draw under the work law with floors `f`. -/
def workK (σ : Fin n → Fin m) (w : Fin m → ℕ) (f : Fin m → ℕ) (K : ℕ) (s : Fin m) : ℕ :=
  workRule K (w s) (stratum σ s).card (totalWork σ w) (f s)

theorem workK_le (σ : Fin n → Fin m) (w : Fin m → ℕ) (f : Fin m → ℕ) (K : ℕ) (s : Fin m) :
    workK σ w f K s ≤ (stratum σ s).card :=
  min_le_left _ _

/-- **The work law**: independent uniform subsets inside the strata of `σ`, stratum `s` drawing `workK σ w f K s`. -/
abbrev work (σ : Fin n → Fin m) (w : Fin m → ℕ) (f : Fin m → ℕ) (K : ℕ) : Law n :=
  stratified σ (workK σ w f K) (workK_le σ w f K)

/-- The count law's rule, the executable's `Flock.Draw.stratumK`: `C · size / population` rounded half up, at least one. -/
def countRule (C size population : ℕ) : ℕ := max 1 ((2 * C * size + population) / (2 * population))

/-- The count law's sizes over `σ`'s strata at count budget `C`: `main`'s stratified law, `stratumK C n_s N`. -/
def countK (σ : Fin n → Fin m) (C : ℕ) (s : Fin m) : ℕ := min (stratum σ s).card (countRule C (stratum σ s).card n)

/-- A law that proves more units than it draws: the same coins, each draw widened by `f`. Closure draws are `f S = S`
with the units `S`'s members read. -/
def widen (L : Law n) (f : Finset (Fin n) → Finset (Fin n)) : Law n where
  Ω := L.Ω
  draw ω := f (L.draw ω)

end Law

section Audits

variable {n m : ℕ} {σ : Fin n → Fin m} {w : Fin m → ℕ} {f : Fin m → ℕ} {K : ℕ}
  {Reg : Type} {session : Finset (Fin n) → Reg → Game Bool}

end Audits

end FlockSoundness.Audit
