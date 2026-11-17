module

public import Definitions.Flock.Game
public import Mathlib.Data.Fin.VecNotation
public import Mathlib.Data.Fintype.Pi
import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.RingTheory.Henselian
import Mathlib.RingTheory.RegularLocalRing.Defs
import Mathlib.RingTheory.SimpleRing.Principal

/-! The `--zk` inner proof's definitions and row lemmas on Mathlib; `inner_sound` is in `InnerSound`. -/

@[expose] public section

namespace FlockSoundness.Discharge.ZkSession

open Game
open scoped ENNReal NNReal

variable {F : Type} [Field F]

/-- An affine form `Σ_i coef_i·h_i + const` in the pads (§4's `Lin`). -/
structure Lin (F : Type) (s : ℕ) where
  coef : Fin s → F
  const : F

def Lin.eval {s : ℕ} (ℓ : Lin F s) (h : Fin s → F) : F := ∑ i, ℓ.coef i * h i + ℓ.const

/-- The pad indices the sacrifice reads: the product `h_ab = h_fa·h_fb` it binds and the sacrificed triple
`x, y, w = x·y` (§2's `fa, fb, hab, tx, ty, tw`). -/
structure Triple (s : ℕ) where
  fa : Fin s
  fb : Fin s
  ab : Fin s
  tx : Fin s
  ty : Fin s
  tw : Fin s

/-- §4's three triple rows at the coin `ε` and the prover's `ρ, σ`:
`ε h_fa + h_tx + ρ`, `h_fb + h_ty + σ`, `ε h_ab + h_tw + σ h_tx + ρ h_ty + ρσ`. -/
def tripleRows {s : ℕ} (tr : Triple s) (ε ρ σ : F) : Fin 3 → Lin F s :=
  ![⟨Pi.single tr.fa ε + Pi.single tr.tx 1, ρ⟩,
    ⟨Pi.single tr.fb 1 + Pi.single tr.ty 1, σ⟩,
    ⟨Pi.single tr.ab ε + Pi.single tr.tw 1 + Pi.single tr.tx σ + Pi.single tr.ty ρ, ρ * σ⟩]

/-- The batch's coefficient vector `c = Σ_k γ_k·coef_k`. -/
def batchCoef {s n : ℕ} (γ : Fin n → F) (R : Fin n → Lin F s) : Fin s → F :=
  fun i => ∑ k, γ k * (R k).coef i

/-- The batch's target `t = −Σ_k γ_k·const_k`, so that `⟨c, h⟩ = t` iff `Σ_k γ_k·R_k(h) = 0`. -/
def batchT {s n : ℕ} (γ : Fin n → F) (R : Fin n → Lin F s) : F := -∑ k, γ k * (R k).const

/-- The Reed–Solomon encoding of a message read as the coefficients of `Σ_i m_i X^i`, at the points `dom` (§3). -/
def enc {N L : ℕ} (dom : Fin N ↪ F) (m : Fin L → F) : Fin N → F :=
  fun j => ∑ i, m i * dom j ^ (i : ℕ)

/-- The first `s` coefficients of a pads message: the pads `h` (or `mu`), without the code's padding. -/
def head {s nP : ℕ} (m : Fin (s + nP) → F) : Fin s → F := fun i => m (Fin.castAdd nP i)

def Agree {N : ℕ} (δ : ℝ≥0) (u v : Fin N → F) : Prop :=
  ∃ S : Finset (Fin N), (1 - δ) * (N : ℝ≥0) ≤ S.card ∧ ∀ j ∈ S, u j = v j

/-- The verifier's final check (§5, step 5): the batched constraint on `y`'s pads, and `C(y)_j = C_h[j] + β·C_mu[j]` at
every queried position. -/
def innerCheck {N : ℕ} (dom : Fin N ↪ F) {s nP q : ℕ} (pos : F → Fin N) {nA : ℕ} (rows : Fin nA → Lin F s)
    (tr : Triple s) (Ch Cμ : Fin N → F) (ε : F) (ρσ : F × F) (γ : Fin (nA + 3) → F) (τ β : F)
    (y : Fin (s + nP) → F) (w : Fin q → F) : Prop :=
  let R := Fin.append rows (tripleRows tr ε ρσ.1 ρσ.2)
  (∑ i, batchCoef γ R i * head y i = batchT γ R + β * τ) ∧
    ∀ j, enc dom y (pos (w j)) = Ch (pos (w j)) + β * Cμ (pos (w j))

instance instDecidableInnerCheckOfDecidableEq [DecidableEq F] {N : ℕ} (dom : Fin N ↪ F) {s nP q : ℕ} (pos : F → Fin N) {nA : ℕ}
    (rows : Fin nA → Lin F s) (tr : Triple s) (Ch Cμ : Fin N → F) (ε : F) (ρσ : F × F) (γ : Fin (nA + 3) → F)
    (τ β : F) (y : Fin (s + nP) → F) (w : Fin q → F) :
    Decidable (innerCheck dom pos rows tr Ch Cμ ε ρσ γ τ β y w) := by
  unfold innerCheck
  infer_instance

def innerGame [Fintype F] [DecidableEq F] {N : ℕ} (dom : Fin N ↪ F) (s nP q : ℕ) (pos : F → Fin N) {nA : ℕ}
    (rows : Fin nA → Lin F s) (tr : Triple s) (Ch Cμ : Fin N → F) : Game Bool := do
  let ε ← draw F
  let ρσ ← recv (F × F)
  let γ ← draw (Fin (nA + 3) → F)
  let τ ← recv F
  let β ← draw F
  let y ← recv (Fin (s + nP) → F)
  let w ← draw (Fin q → F)
  pure (decide (innerCheck dom pos rows tr Ch Cμ ε ρσ γ τ β y w))

def InnerGood [DecidableEq F] {N : ℕ} (dom : Fin N ↪ F) (s nP : ℕ) {nA : ℕ} (rows : Fin nA → Lin F s) (tr : Triple s)
    (δ : ℝ≥0) (Ch : Fin N → F) : Prop :=
  ∃ m : Fin (s + nP) → F, Agree δ Ch (enc dom m) ∧ (∀ k, (rows k).eval (head m) = 0) ∧
    head m tr.ab = head m tr.fa * head m tr.fb

/-- The inner proof's soundness error: BCIKS20's `N/|F|` (or the sacrifice, the batch and `β`, `1/|F|` each) plus
`q` queries at distance `δ`. -/
noncomputable def εInner (N : ℕ) (F : Type) [Fintype F] (q : ℕ) (δ : ℝ≥0) : ℝ≥0∞ :=
  (N + 3 : ℝ≥0∞) / Fintype.card F + ((1 - δ : ℝ≥0) : ℝ≥0∞) ^ q

theorem sum_single_mul {s : ℕ} (i : Fin s) (a : F) (h : Fin s → F) :
    ∑ j, (Pi.single i a : Fin s → F) j * h j = a * h i := by
  rw [Finset.sum_eq_single i (fun j _ hj => by simp [hj]) (by simp)]
  simp

theorem tripleRows_eval_zero {s : ℕ} (tr : Triple s) (ε ρ σ : F) (h : Fin s → F) :
    (tripleRows tr ε ρ σ 0).eval h = ε * h tr.fa + h tr.tx + ρ := by
  simp only [tripleRows, Lin.eval, Matrix.cons_val_zero, Pi.add_apply, add_mul,
    Finset.sum_add_distrib, sum_single_mul, one_mul]

theorem tripleRows_eval_one {s : ℕ} (tr : Triple s) (ε ρ σ : F) (h : Fin s → F) :
    (tripleRows tr ε ρ σ 1).eval h = h tr.fb + h tr.ty + σ := by
  simp only [tripleRows, Lin.eval, Matrix.cons_val_one, Matrix.cons_val_zero, Pi.add_apply,
    add_mul, Finset.sum_add_distrib, sum_single_mul, one_mul]

theorem tripleRows_eval_two {s : ℕ} (tr : Triple s) (ε ρ σ : F) (h : Fin s → F) :
    (tripleRows tr ε ρ σ 2).eval h =
      ε * h tr.ab + h tr.tw + σ * h tr.tx + ρ * h tr.ty + ρ * σ := by
  simp [tripleRows, Lin.eval, add_mul, Finset.sum_add_distrib, sum_single_mul]

theorem sacrifice_eq {s : ℕ} (tr : Triple s) (ε ρ σ : F) (h : Fin s → F)
    (h0 : (tripleRows tr ε ρ σ 0).eval h = 0) (h1 : (tripleRows tr ε ρ σ 1).eval h = 0) :
    (tripleRows tr ε ρ σ 2).eval h =
      ε * (h tr.ab + h tr.fa * h tr.fb) + (h tr.tw - h tr.tx * h tr.ty) := by
  rw [tripleRows_eval_zero] at h0
  rw [tripleRows_eval_one] at h1
  rw [tripleRows_eval_two]
  linear_combination (h tr.ty + σ) * h0 - ε * h tr.fa * h1

theorem batch_eval {s n : ℕ} (γ : Fin n → F) (R : Fin n → Lin F s) (h : Fin s → F) :
    ∑ i, batchCoef γ R i * h i - batchT γ R = ∑ k, γ k * (R k).eval h := by
  simp only [batchCoef, batchT, Lin.eval, Finset.sum_mul, sub_neg_eq_add, mul_add,
    Finset.sum_add_distrib, Finset.mul_sum]
  rw [Finset.sum_comm]
  congr 1
  refine Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun i _ => ?_
  ring

end FlockSoundness.Discharge.ZkSession
