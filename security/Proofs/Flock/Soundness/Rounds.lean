module

public import Definitions.Core.Game
public import Definitions.Flock.Piop
public import Mathlib.Algebra.Polynomial.Degree.Defs
public import Mathlib.Algebra.Polynomial.Eval.Defs
public import Proofs.Core.Game.Poly
import Mathlib.Algebra.Order.Algebra
import Mathlib.Algebra.Polynomial.Degree.Lemmas
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.Positivity.Finset

/-! One sumcheck round, in each of the spec's three encodings, keeps a false claim false except with probability `2/|F|`
(characteristic 2). -/

@[expose] public section

namespace FlockSoundness.Rounds

open Polynomial Game Model

open scoped ENNReal

variable {F : Type} [Field F] [Fintype F] [DecidableEq F] [CharP F 2]

omit [Fintype F] [DecidableEq F] in
theorem two_eq_zero : (2 : F) = 0 := by exact_mod_cast CharP.cast_eq_zero F 2

omit [Fintype F] [DecidableEq F] in
theorem add_self (a : F) : a + a = 0 := by
  rw [← two_mul, two_eq_zero, zero_mul]

noncomputable def residualPoly (req claim g₁ gInf : F) : F[X] :=
  C ((claim + req * g₁) * (1 + req)⁻¹) * (1 + X) + C g₁ * X + C gInf * X * (1 + X)

omit [Fintype F] [DecidableEq F] [CharP F 2] in
theorem residualStep_eq (req claim g₁ gInf ρ : F) :
    residualStep req claim g₁ gInf ρ = (residualPoly req claim g₁ gInf).eval ρ := by
  simp only [residualStep, residualPoly, eval_add, eval_mul, eval_C, eval_X, eval_one]

omit [Fintype F] [DecidableEq F] [CharP F 2] in
theorem residualPoly_natDegree_le (req claim g₁ gInf : F) :
    (residualPoly req claim g₁ gInf).natDegree ≤ 2 := by
  unfold residualPoly
  have h1 : (C ((claim + req * g₁) * (1 + req)⁻¹) * (1 + X) : F[X]).natDegree ≤ 2 :=
    (natDegree_C_mul_le _ _).trans ((natDegree_add_le _ _).trans (by simp))
  have h2 : (C g₁ * X : F[X]).natDegree ≤ 2 := (natDegree_C_mul_le _ _).trans (by simp)
  have h3 : (C gInf * X * (1 + X) : F[X]).natDegree ≤ 2 := by
    refine (natDegree_mul_le).trans ?_
    have hx : (C gInf * X : F[X]).natDegree ≤ 1 := (natDegree_C_mul_le _ _).trans (by simp)
    have hy : ((1 : F[X]) + X).natDegree ≤ 1 := (natDegree_add_le _ _).trans (by simp)
    omega
  exact (natDegree_add_le _ _).trans (max_le ((natDegree_add_le _ _).trans (max_le h1 h2)) h3)

theorem residualRound_sound (req claim : F) (hreq : req ≠ 1) (Gt : F[X])
    (hGt : Gt.natDegree ≤ 2) (hclaim : claim ≠ (1 + req) * Gt.eval 0 + req * Gt.eval 1)
    (g₁ gInf : F) :
    prCoin (fun ρ : F => residualStep req claim g₁ gInf ρ = Gt.eval ρ) ≤
      2 / Fintype.card F := by
  have h1req : (1 : F) + req ≠ 0 := by
    intro h
    apply hreq
    have : req = -1 := by linear_combination h
    rw [this, neg_eq_iff_eq_neg, ← sub_eq_zero]
    have := add_self (1 : F)
    linear_combination this
  have hne : residualPoly req claim g₁ gInf ≠ Gt := by
    intro h
    apply hclaim
    have e0 : Gt.eval 0 = (claim + req * g₁) * (1 + req)⁻¹ := by
      rw [← h]; simp [residualPoly]
    have e1 : Gt.eval 1 = g₁ := by
      rw [← h]
      simp only [residualPoly, eval_add, eval_mul, eval_C, eval_X, eval_one]
      rw [show (1 : F) + 1 = 0 from add_self 1]
      ring
    rw [e0, e1, mul_comm (1 + req), mul_assoc, inv_mul_cancel₀ h1req, mul_one]
    have := add_self (req * g₁)
    linear_combination -this
  simp_rw [residualStep_eq]
  exact Verity.Game.prCoin_eval_eq_eval_le _ _ hne 2 (residualPoly_natDegree_le _ _ _ _) hGt

omit [CharP F 2] in
theorem sumRound_sound (claim : F) (P T : F[X]) (hP : P.natDegree ≤ 2) (hT : T.natDegree ≤ 2)
    (hsum : P.eval 0 + P.eval 1 = claim) (hclaim : claim ≠ T.eval 0 + T.eval 1) :
    prCoin (fun t : F => P.eval t = T.eval t) ≤ 2 / Fintype.card F := by
  have hne : P ≠ T := by
    intro h; apply hclaim; rw [← hsum, h]
  exact Verity.Game.prCoin_eval_eq_eval_le P T hne 2 hP hT

noncomputable def lincheckPoly (claim q₁ qInf : F) : F[X] :=
  C (claim + q₁) + C (claim + q₁ + q₁ + qInf) * X + C qInf * X ^ 2

omit [Fintype F] [DecidableEq F] in
theorem lincheckPoly_sum (claim q₁ qInf : F) :
    (lincheckPoly claim q₁ qInf).eval 0 + (lincheckPoly claim q₁ qInf).eval 1 = claim := by
  simp only [lincheckPoly, eval_add, eval_mul, eval_C, eval_X, eval_pow, zero_pow two_ne_zero,
    mul_zero, add_zero, one_pow, mul_one]
  have h1 := add_self (claim + q₁)
  have h2 := add_self q₁
  have h3 := add_self qInf
  linear_combination h1 + h2 + h3

omit [Fintype F] [DecidableEq F] [CharP F 2] in
theorem lincheckPoly_natDegree_le (claim q₁ qInf : F) :
    (lincheckPoly claim q₁ qInf).natDegree ≤ 2 := by
  unfold lincheckPoly
  refine (natDegree_add_le _ _).trans (max_le ((natDegree_add_le _ _).trans (max_le ?_ ?_)) ?_)
  · simp
  · exact (natDegree_C_mul_le _ _).trans (by simp)
  · exact (natDegree_C_mul_le _ _).trans (by simp)

omit [Fintype F] [DecidableEq F] [CharP F 2] in
theorem lincheckStep_eq (claim q₁ qInf t : F) :
    qInf * t ^ 2 + (claim + q₁ + q₁ + qInf) * t + (claim + q₁) =
      (lincheckPoly claim q₁ qInf).eval t := by
  simp only [lincheckPoly, eval_add, eval_mul, eval_C, eval_X, eval_pow]; ring

theorem lincheckRound_sound (claim : F) (T : F[X]) (hT : T.natDegree ≤ 2)
    (hclaim : claim ≠ T.eval 0 + T.eval 1) (q₁ qInf : F) :
    prCoin (fun t : F => qInf * t ^ 2 + (claim + q₁ + q₁ + qInf) * t + (claim + q₁) = T.eval t) ≤
      2 / Fintype.card F := by
  simp_rw [lincheckStep_eq]
  exact sumRound_sound claim _ T (lincheckPoly_natDegree_le _ _ _) hT (lincheckPoly_sum _ _ _)
    hclaim

end FlockSoundness.Rounds
