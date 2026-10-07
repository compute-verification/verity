import Proofs.Flock.Soundness.Game.Poly
import Proofs.Flock.Soundness.Model.Piop
import Proofs.Flock.Soundness.Model.Ligerito
import Mathlib.Algebra.CharP.Defs

/-!
# One sumcheck round

The spec encodes a round's quadratic in three ways (PROTOCOL.md §9.2 Z6, §10 L3, §13.2). In each,
the verifier derives a quadratic from the prover's message and the running claim, and the new
claim is its value at the fresh coin. If the running claim is false, the derived quadratic
differs from the true one whatever the prover sends, so the new claim is false except with
probability `2/|F|` (`Game.prCoin_eval_eq_eval_le`). These are the round-by-round lemmas the
zerocheck, lincheck and Ligerito phases of `rep_sound` chain; a Fiat–Shamir proof would reuse
them unchanged (`DESIGN.md` §8).

All three identities use characteristic 2 (`1 + 1 = 0`).
-/

namespace FlockSoundness.Rounds

open Polynomial Game Model

open scoped ENNReal

variable {F : Type} [Field F] [Fintype F] [DecidableEq F] [CharP F 2]

omit [Fintype F] [DecidableEq F] in
theorem two_eq_zero : (2 : F) = 0 := by exact_mod_cast CharP.cast_eq_zero F 2

omit [Fintype F] [DecidableEq F] in
theorem add_self (a : F) : a + a = 0 := by
  rw [← two_mul, two_eq_zero, zero_mul]

/-! ## The residual round of the zerocheck (Z6) -/

/-- The verifier's quadratic in a residual round: `G(X) = g₀(1 + X) + g₁X + g∞X(1 + X)`, with
`g₀ = (claim + r_eq·g₁)/(1 + r_eq)` (§9.2). -/
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

/-- **Residual round.** If the running claim is not `(1 + r_eq)·G(0) + r_eq·G(1)` for the true
eq-stripped quadratic `G`, and `r_eq ≠ 1`, then whatever `(G(1), G(∞))` the prover sends, the new
claim equals the true `G(ρ)` with probability at most `2/|F|`. -/
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
  exact prCoin_eval_eq_eval_le _ _ hne 2 (residualPoly_natDegree_le _ _ _ _) hGt

/-! ## Rounds that impose `q(0) + q(1) = claim` (L3 and Ligerito) -/

omit [CharP F 2] in
/-- **A round whose polynomial sums to the claim.** If the verifier's quadratic `P` satisfies
`P(0) + P(1) = claim` by construction, and the claim is not `T(0) + T(1)` for the true quadratic
`T`, then `P(t) = T(t)` at the fresh coin with probability at most `2/|F|`. -/
theorem sumRound_sound (claim : F) (P T : F[X]) (hP : P.natDegree ≤ 2) (hT : T.natDegree ≤ 2)
    (hsum : P.eval 0 + P.eval 1 = claim) (hclaim : claim ≠ T.eval 0 + T.eval 1) :
    prCoin (fun t : F => P.eval t = T.eval t) ≤ 2 / Fintype.card F := by
  have hne : P ≠ T := by
    intro h; apply hclaim; rw [← hsum, h]
  exact prCoin_eval_eq_eval_le P T hne 2 hP hT

/-- The lincheck's quadratic from `(q(1), q(∞))` (§10 L3): `q(X) = e₀ + c₁X + q∞X²`, with
`e₀ = claim + q(1)` and `c₁ = e₀ + q(1) + q(∞)`. -/
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
/-- The lincheck round's new claim is the quadratic at the coin (as `lincheckRounds` computes). -/
theorem lincheckStep_eq (claim q₁ qInf t : F) :
    qInf * t ^ 2 + (claim + q₁ + q₁ + qInf) * t + (claim + q₁) =
      (lincheckPoly claim q₁ qInf).eval t := by
  simp only [lincheckPoly, eval_add, eval_mul, eval_C, eval_X, eval_pow]; ring

/-- **Lincheck round.** A false running claim stays false at `t` except with probability `2/|F|`. -/
theorem lincheckRound_sound (claim : F) (T : F[X]) (hT : T.natDegree ≤ 2)
    (hclaim : claim ≠ T.eval 0 + T.eval 1) (q₁ qInf : F) :
    prCoin (fun t : F => qInf * t ^ 2 + (claim + q₁ + q₁ + qInf) * t + (claim + q₁) = T.eval t) ≤
      2 / Fintype.card F := by
  simp_rw [lincheckStep_eq]
  exact sumRound_sound claim _ T (lincheckPoly_natDegree_le _ _ _) hT (lincheckPoly_sum _ _ _)
    hclaim

end FlockSoundness.Rounds
