module

import all Init.Data.Array.Basic
import all Init.Data.Range.Basic
import all Init.Data.Nat.Power2.Basic
public import Proofs.Flock.Claims.DeployPlainCode
public import Proofs.Flock.Claims.ZkDecodeFlock
public import Proofs.Flock.Soundness.Accounting.PaddedNumbers
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

/-! The radius `ε_table` is computed at, and the decoder's. -/

@[expose] public section

namespace Claims.Flock.Numbers

open FlockSoundness FlockSoundness.Accounting

variable {wideLeaf : Bool}

theorem level_rate_eq (l : Level) : l.rate = (2 : ℝ) ^ l.logCols / 2 ^ l.logLen := by
  rw [Level.rate, Level.logLen, pow_add]
  field_simp
  rw [← mul_pow]
  norm_num

theorem level_radius_eq (l : Level) :
    l.radius = 1 - Real.sqrt ((2 : ℝ) ^ l.logCols / 2 ^ l.logLen) - 1 / 200 := by
  rw [Level.radius, level_rate_eq, eta]

theorem level_padRadius_eq (l : Level) (t : ℕ) :
    l.padRadius t = 1 - Real.sqrt (((2 : ℝ) ^ l.logCols + t) / 2 ^ l.logLen) - 1 / 200 := by
  rw [Level.padRadius, Level.padRate, eta]
  push_cast
  rfl

theorem plain_level₀ : ∀ i : PlainIdx,
    ((Accounting.fast100 i.m).getD default).k0 = plainLanes i ∧
      ((Accounting.fast100 i.m).getD default).level₀.logCols = plainCols i ∧
      ((Accounting.fast100 i.m).getD default).level₀.logLen = plainLen i ∧
      ((Accounting.fast100 i.m).getD default).level₀.logInvRate = 1 := by
  decide +kernel

theorem plain_radius_eq (i : PlainIdx) :
    ((Accounting.fast100 i.m).getD default).level₀.radius =
      1 - Real.sqrt ((2 : ℝ) ^ plainCols i / 2 ^ plainLen i) - 1 / 200 := by
  obtain ⟨-, hc, hl, -⟩ := plain_level₀ i
  rw [level_radius_eq, hc, hl]

theorem plain_rate_half (i : PlainIdx) : (2 : ℝ) ^ plainCols i / 2 ^ plainLen i = 1 / 2 := by
  rw [(plainShape_table i).2.2.2.1, pow_succ]
  field_simp

theorem plain_radius_half (i : PlainIdx) :
    ((Accounting.fast100 i.m).getD default).level₀.radius = 1 - Real.sqrt (1 / 2) - 1 / 200 := by
  rw [plain_radius_eq, plain_rate_half]

theorem radius_half_bounds :
    (0.2878 : ℝ) < 1 - Real.sqrt (1 / 2) - 1 / 200 ∧ 1 - Real.sqrt (1 / 2) - 1 / 200 < 0.2879 := by
  have h1 : (0.7071 : ℝ) < Real.sqrt (1 / 2) := by
    rw [Real.lt_sqrt (by norm_num)]
    norm_num
  have h2 : Real.sqrt (1 / 2) < 0.7072 := by
    rw [Real.sqrt_lt' (by norm_num)]
    norm_num
  constructor <;> linarith

theorem zk_level₀ : ∀ i : ZkIdx,
    (m1Schedule i.wideLeaf i.m).k0 = zkLanes i ∧ (m1Schedule i.wideLeaf i.m).level₀.logCols = zkCols i ∧
      (m1Schedule i.wideLeaf i.m).level₀.logLen = zkLen i ∧ m1Pad i.wideLeaf i.m = zkPad i ∧
      (m1Schedule i.wideLeaf i.m).level₀.queries = (zkLevel0 i.wideLeaf i.m).q := by
  decide +kernel

theorem zk_radius_eq (i : ZkIdx) :
    (m1Schedule i.wideLeaf i.m).level₀.padRadius (m1Pad i.wideLeaf i.m) =
      1 - Real.sqrt (((2 : ℝ) ^ zkCols i + zkPad i) / 2 ^ zkLen i) - 1 / 200 := by
  obtain ⟨-, hc, hl, ht, -⟩ := zk_level₀ i
  rw [level_padRadius_eq, hc, hl, ht]

theorem zkWidth_eq (i : ZkIdx) : zkWidth i = 2 ^ (m1Schedule i.wideLeaf i.m).k0 := by
  rw [(zk_level₀ i).1]

theorem m1SqrtUp_le : ∀ m, 25 ≤ m → m ≤ 35 → m1SqrtUp wideLeaf m ≤ 377 / 500 := by
  intro m h1 h2
  cases wideLeaf <;> interval_cases m <;> decide +kernel

theorem zk_radius_ge (i : ZkIdx) : (241 : ℝ) / 1000 ≤ (m1Schedule i.wideLeaf i.m).level₀.padRadius (m1Pad i.wideLeaf i.m) := by
  obtain ⟨h1, h2⟩ := ZkIdx.m_range i
  have hp := m1_padNumOK (wide := i.wideLeaf) i.m h1 h2
  have hs : Real.sqrt ((m1Schedule i.wideLeaf i.m).level₀.padRate (m1Pad i.wideLeaf i.m)) ≤ (m1SqrtUp i.wideLeaf i.m : ℝ) :=
    Real.sqrt_le_iff.2 ⟨by exact_mod_cast hp.h0, hp.hs⟩
  have hu := (Rat.cast_le (K := ℝ)).2 (m1SqrtUp_le (wideLeaf := i.wideLeaf) i.m h1 h2)
  push_cast at hu
  rw [Level.padRadius, eta]
  linarith

theorem zkPad_pos : ∀ i : ZkIdx, 1 ≤ zkPad i := by
  decide +kernel

theorem zk_radius_lt_plain (i : ZkIdx) :
    (m1Schedule i.wideLeaf i.m).level₀.padRadius (m1Pad i.wideLeaf i.m) < 1 - Real.sqrt (1 / 2) - 1 / 200 := by
  rw [zk_radius_eq]
  have ht : (1 : ℝ) ≤ zkPad i := by exact_mod_cast zkPad_pos i
  have hr : (1 : ℝ) / 2 < ((2 : ℝ) ^ zkCols i + zkPad i) / 2 ^ zkLen i := by
    rw [(zkShape_table i).2.2.2.1, pow_succ, lt_div_iff₀ (by positivity)]
    linarith
  have := Real.sqrt_lt_sqrt (by norm_num) hr
  linarith

end Claims.Flock.Numbers
