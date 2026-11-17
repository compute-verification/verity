module

public import Proofs.Flock.Claims.Decode
public import Proofs.Flock.Claims.Numbers
public import Proofs.Flock.Soundness.Accounting.PaddedNumbers
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Algebra.Order.Star.Real
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Data.Rat.Star
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
import Std.Tactic.BVDecide.Normalize.Prop
import VCVio.Prelude.Core

/-! `nMin` at the deployed parameters: one run opens fewer level-0 columns, and `E^rep`'s accepted reruns more. -/

@[expose] public section

namespace Claims.Flock.Numbers

open FlockSoundness.Accounting

variable {wideLeaf : Bool}

noncomputable def agree₀ : ℝ := Real.sqrt (1 / 2) + 1 / 200

theorem sqrt_half_bounds : (0.7071067811 : ℝ) < Real.sqrt (1 / 2) ∧ Real.sqrt (1 / 2) < 0.7071067812 := by
  constructor
  · rw [Real.lt_sqrt (by norm_num)]
    norm_num
  · rw [Real.sqrt_lt' (by norm_num)]
    norm_num

theorem agree₀_pos : 0 < agree₀ := by
  unfold agree₀
  positivity

theorem agree₀_le_one : agree₀ ≤ 1 := by
  unfold agree₀
  linarith [sqrt_half_bounds.2]

theorem agree₀_sq_bounds : (0.507096067811 : ℝ) ≤ agree₀ ^ 2 ∧ agree₀ ^ 2 ≤ 0.507096067812 := by
  obtain ⟨h1, h2⟩ := sqrt_half_bounds
  have hs : Real.sqrt (1 / 2) ^ 2 = 1 / 2 := Real.sq_sqrt (by norm_num)
  have e : agree₀ ^ 2 = 1 / 2 + Real.sqrt (1 / 2) / 100 + 1 / 40000 := by
    unfold agree₀
    nlinarith
  rw [e]
  constructor <;> linarith

theorem johnsonCount_eq_of {k J : ℕ} {a lo hi : ℝ} (hJ : 1 ≤ J) (hlo0 : 0 < lo) (hlo : lo ≤ a ^ 2)
    (hhi : a ^ 2 ≤ hi) (h1 : ((J : ℝ) - 1) * hi ≤ k + 1) (h2 : (k : ℝ) + 1 < J * lo) : johnsonCount k a = J := by
  unfold johnsonCount
  have ha2 : 0 < a ^ 2 := lt_of_lt_of_le hlo0 hlo
  have hJ1 : (1 : ℝ) ≤ J := by exact_mod_cast hJ
  have hfl : ⌊((k : ℝ) + 1) / a ^ 2⌋₊ = J - 1 := by
    rw [Nat.floor_eq_iff (by positivity), Nat.cast_sub hJ, Nat.cast_one, le_div_iff₀ ha2, div_lt_iff₀ ha2]
    constructor
    · nlinarith
    · simp only [sub_add_cancel]
      nlinarith
  omega

def nMinVal : ℕ → ℕ
  | 9 => 1012
  | 10 => 2022
  | 11 => 4041
  | 12 => 8080
  | 13 => 16157
  | 14 => 32312
  | 17 => 258478
  | 18 => 516954
  | 19 => 1033905
  | 20 => 2067808
  | 21 => 4135613
  | 22 => 8271224
  | _ => 0

theorem johnsonCount_agree₀ {c : ℕ} (hc : c ∈ [9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22]) :
    johnsonCount (2 ^ c) agree₀ = nMinVal c := by
  obtain ⟨hlo, hhi⟩ := agree₀_sq_bounds
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hc
  rcases hc with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    exact johnsonCount_eq_of (by simp [nMinVal]) (by norm_num) hlo hhi (by norm_num [nMinVal])
      (by norm_num [nMinVal])

theorem fast100_logCols₀ {m : ℕ} {sch : Schedule} (h : fast100 m = some sch) :
    (sch.levels.headD default).logCols ∈ [9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22] ∧
      (sch.levels.headD default).logLen = (sch.levels.headD default).logCols + 1 := by
  have hm : 22 ≤ m ∧ m ≤ 35 := by
    by_contra hc
    have : fast100 m = none := by unfold fast100; split <;> simp_all
    simp [this] at h
  obtain ⟨h1, h2⟩ := hm
  interval_cases m <;> (simp only [fast100, Option.some.injEq] at h; subst h; decide)

theorem nMin_fast100 {m : ℕ} {sch : Schedule} (h : fast100 m = some sch) :
    johnsonCount (2 ^ (sch.levels.headD default).logCols) agree₀ = nMinVal (sch.levels.headD default).logCols :=
  johnsonCount_agree₀ (fast100_logCols₀ h).1

theorem nMin_lt_len {c : ℕ} (hc : c ∈ [9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22]) :
    nMinVal c < 2 ^ (c + 1) ∧ 2 ^ (c + 1) - nMinVal c + 1 ≥ 2 ^ (c + 1) / 80 := by
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hc
  rcases hc with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide

theorem queries_lt_nMin {c q : ℕ} (hc : c ∈ [9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22]) (hq : q ≤ 277) :
    q < nMinVal c := by
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hc
  rcases hc with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> simp [nMinVal] <;> omega

theorem len_lt_k_mul_queries {d : ℕ} (hd : d ∈ [10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23]) :
    2 ^ d < 218 * kRep d := by
  rw [kRep_eq hd]
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hd
  rcases hd with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide

theorem miss_ge {c q : ℕ} (hc : c ∈ [9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22]) (hq : q ≤ 277) :
    (1 / 64 : ℚ) ≤ (((nMinVal c - 1 : ℕ) : ℚ) / 2 ^ (c + 1)) ^ q := by
  have hle : ∀ c, c ∈ [9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22] →
      (1 / 64 : ℚ) ≤ (((nMinVal c - 1 : ℕ) : ℚ) / 2 ^ (c + 1)) ^ 277 := by
    intro c hc
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hc
    rcases hc with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide +kernel
  refine (hle c hc).trans (pow_le_pow_of_le_one (by positivity) ?_ hq)
  rw [div_le_one (by positivity)]
  have := (nMin_lt_len hc).1
  exact_mod_cast (Nat.sub_le _ _).trans this.le

theorem zk_nMin_gt_len {m : ℕ} (hm : m ∈ [25, 26, 27]) :
    2 ^ (((m1Schedule wideLeaf m).levels.headD default).logCols + 1) <
      johnsonCount (2 ^ ((m1Schedule wideLeaf m).levels.headD default).logCols + m1Pad wideLeaf m) agree₀ := by
  obtain ⟨hlo, hhi⟩ := agree₀_sq_bounds
  have key : ∀ w : Bool,
      (((m1Schedule w 25).levels.headD default).logCols = 12 ∧ m1Pad w 25 = 554) ∧
      (((m1Schedule w 26).levels.headD default).logCols = 13 ∧ m1Pad w 26 = 484) ∧
      (((m1Schedule w 27).levels.headD default).logCols = 14 ∧ m1Pad w 27 = 458) := by
    intro w; cases w <;> decide
  obtain ⟨⟨c25, p25⟩, ⟨c26, p26⟩, c27, p27⟩ := key wideLeaf
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hm
  rcases hm with rfl | rfl | rfl
  · rw [c25, p25,
      johnsonCount_eq_of (J := 9172) (by norm_num) (by norm_num) hlo hhi (by norm_num) (by norm_num)]
    norm_num
  · rw [c26, p26,
      johnsonCount_eq_of (J := 17112) (by norm_num) (by norm_num) hlo hhi (by norm_num) (by norm_num)]
    norm_num
  · rw [c27, p27,
      johnsonCount_eq_of (J := 33215) (by norm_num) (by norm_num) hlo hhi (by norm_num) (by norm_num)]
    norm_num

end Claims.Flock.Numbers
