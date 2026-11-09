module

public import Definitions.Accounting.TS.Routes
public import Mathlib.Analysis.Complex.ExponentialBounds
public import Mathlib.Analysis.SpecialFunctions.Log.Deriv
public import Proofs.Accounting.Headline.Whole
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-!
# `ε_jl` at `p = 7/8` with `ln(8/7) ≥ 2/15`

`epsJl_le` bounds the fold luck's term `ε_jl` at `p = 7/8`, `48,000` rounds and `δ′ ≥ 2^-41` by
`κ·41·0.6931471808/6400`, from `ln 2 < 0.6931471808` and `ln(8/7) ≥ 2/15`, the first term of
`ln(1 + 1/a) = Σ_k 2/(2k + 1)·(2a + 1)^-(2k+1)` at `a = 7`. sm_120's measured headline (`Headline/Sm120Measured.lean`)
and the data-centre headlines' folded branch (`HeadlineDc/Branches.lean`) read it.
-/

@[expose] public section

namespace Accounting.SecurityProofs.TS.Headline

open Accounting.TS Accounting.Assumptions.TS Pouw.RelativeSecurity Accounting.SecurityProofs.TS

/-- `ε_jl ≤ κ·41·0.6931471808/6400` at `p = 7/8`, `48,000` rounds and `δ′ ≥ 2^-41`, from `ln 2 < 0.6931471808` and
`ln(8/7) ≥ 2/15`, the first term of `ln(1 + 1/a) = Σ_k 2/(2k + 1)·(2a + 1)^-(2k+1)` at `a = 7`. -/
theorem epsJl_le {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F]
    [Fintype X] (Z : Model K R T L U F X) (Fl : Labelled K) (lam : R → ℝ) (hκ : 0 ≤ Z.kappa Fl lam) (S : ℕ)
    (hS : (S : ℝ) = 48000) (δ' : ℝ) (hδ' : (1 / 2 : ℝ) ^ 41 ≤ δ') :
    Z.epsJl Fl lam (7 / 8) δ' S ≤ Z.kappa Fl lam * (41 * (6931471808 / 10000000000) / 6400) := by
  have hlog2 : Real.log 2 < 6931471808 / 10000000000 := lt_of_lt_of_eq Real.log_two_lt_d9 (by norm_num)
  have hδpos : 0 < δ' := lt_of_lt_of_le (by positivity) hδ'
  have hinv : 1 / δ' ≤ (2 : ℝ) ^ 41 := by
    have h := one_div_le_one_div_of_le (by positivity : (0 : ℝ) < (1 / 2) ^ 41) hδ'
    rwa [show (1 : ℝ) / (1 / 2) ^ 41 = 2 ^ 41 by norm_num] at h
  have hlogδ : Real.log (1 / δ') ≤ 41 * (6931471808 / 10000000000) := by
    calc Real.log (1 / δ') ≤ Real.log ((2 : ℝ) ^ 41) := Real.log_le_log (one_div_pos.mpr hδpos) hinv
      _ = 41 * Real.log 2 := by rw [Real.log_pow]; norm_num
      _ ≤ 41 * (6931471808 / 10000000000) := by linarith
  have hlogp : (2 : ℝ) / 15 ≤ Real.log (1 / (7 / 8)) := by
    have h0 := le_hasSum (Real.hasSum_log_one_add_inv (by norm_num : (0 : ℝ) < 7)) 0 fun j _ => by
      first
        | positivity
        | (dsimp only; positivity)
    rw [show (1 : ℝ) / (7 / 8) = 1 + 7⁻¹ by norm_num]
    refine Eq.trans_le ?_ h0
    norm_num
  have hq : Real.log (1 / δ') / (Real.log (1 / (7 / 8)) * 48000) ≤ 41 * (6931471808 / 10000000000) / 6400 :=
    div_le_div₀ (by norm_num) hlogδ (by norm_num) (by linarith)
  unfold Model.epsJl
  rw [hS]
  exact (mul_div_assoc _ _ _).trans_le (mul_le_mul_of_nonneg_left hq hκ)

end Accounting.SecurityProofs.TS.Headline
