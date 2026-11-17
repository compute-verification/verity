module

public import Proofs.Flock.Claims.GS.RadiusFamily
public import Proofs.Flock.Claims.DeployPlainCode
import all Init.Data.Array.Basic
import all Init.Data.Range.Basic
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
import Mathlib.Tactic.Polynomial.Basic
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init

/-! The radius decoders the deployed instances name (`plainDecodeRD`, `zkDecodeRD`), read off the codes and not the
records. -/

@[expose] public section

namespace Claims.GS

open Polynomial FlockSoundness FlockSoundness.Model FlockLevel3

/-- The plain reader at index `i` (`m = 22 + i`). -/
noncomputable def plainReader (i : Flock.PlainIdx) : Reader (Flock.flockPlainLevelZero i) :=
  levelZeroReader execArith execArith_correct (Flock.plainLanes i) (Flock.plainCols i) (Flock.plainCols_le i)

/-- **The plain radius decoder's agreement** `√ρ + 1/200`, at the plain level-0 rate `ρ = 2^cols/2^d`. -/
noncomputable def plainAgreeR (i : Flock.PlainIdx) : ℝ :=
  Real.sqrt ((2 : ℝ) ^ Flock.plainCols i / 2 ^ Flock.plainLen i) + 1 / 200

theorem plainAgreeR_pos (i : Flock.PlainIdx) : 0 < plainAgreeR i := by
  unfold plainAgreeR
  positivity

theorem plainShape_nk : ∀ i : Flock.PlainIdx, 10000 < 2 ^ Flock.plainLen i * 2 ^ Flock.plainCols i := by
  decide +kernel

theorem plainJohnson_le_len (i : Flock.PlainIdx) :
    johnsonCount (2 ^ Flock.plainCols i) (plainAgreeR i) ≤ 2 ^ Flock.plainLen i := by
  have := johnsonCount_le_len (k := 2 ^ Flock.plainCols i) (n := 2 ^ Flock.plainLen i) (by positivity)
    (plainShape_nk i)
  push_cast at this
  exact this

/-- **The plain radius decoder** at `m = 22 + i`: the zero-filled word of length `n = 2^d`, then the counted
Guruswami–Sudan decoder on its `n` points at agreement `√ρ + 1/200`. -/
noncomputable def plainDecodeR (i : Flock.PlainIdx) (os : List (ℕ × (Fin (2 ^ Flock.plainLanes i) → GF128))) :
    List (Message GF128 (Flock.plainLanes i) (Flock.plainCols i)) :=
  gsDecode (Flock.flockPlainLevelZero i) rf128 (plainReader i) (plainAgreeR i) (fillPts (2 ^ Flock.plainLen i) os)

/-- `plainDecodeR` with its count: the fill step's (`fillPtsC`), then the decoder's on the `n` points, one step per
point. -/
noncomputable def plainCountedR (i : Flock.PlainIdx) (os : List (ℕ × (Fin (2 ^ Flock.plainLanes i) → GF128))) :
    List (Message GF128 (Flock.plainLanes i) (Flock.plainCols i)) × ℕ :=
  ((gsCounted (Flock.flockPlainLevelZero i) rf128 (plainReader i) (plainAgreeR i) 1
      (fillPtsC (2 ^ Flock.plainLen i) os).1).1,
    (fillPtsC (2 ^ Flock.plainLen i) os).2 +
      (gsCounted (Flock.flockPlainLevelZero i) rf128 (plainReader i) (plainAgreeR i) 1
        (fillPtsC (2 ^ Flock.plainLen i) os).1).2)

theorem listWithin_plainDecodeR (i : Flock.PlainIdx) :
    Flock.ListWithin (fun _ => 2 ^ Flock.plainLen i)
      (Flock.levelZeroEnc execArith (Flock.plainLanes i) (Flock.plainCols i))
      (fun _ => 1 - Real.sqrt ((2 : ℝ) ^ Flock.plainCols i / 2 ^ Flock.plainLen i) - 1 / 200)
      (plainDecodeR i) := by
  intro os M hM
  have hlt : ∀ p ∈ fillPts (2 ^ Flock.plainLen i) os, p.1 < 2 ^ 64 := fun p hp =>
    lt_of_lt_of_le (lt_of_mem_fillPts hp)
      (Nat.pow_le_pow_right (by norm_num) ((Flock.plainShape_table i).2.2.2.2.1.trans (by norm_num)))
  refine mem_gsDecode (Flock.flockPlainLevelZero i) rf128 (plainReader i) (plainAgreeR_pos i) _
    (Flock.levelZero_points_nodup execArith execArith_correct _ _ _ _ (nodup_fillPts _ os) hlt)
    (by rw [length_fillPts]; exact plainJohnson_le_len i) M ?_
  unfold IRS.agreeCount
  rw [Flock.flockPlainLevelZero, Flock.levelZero_enc, length_fillPts, length_filter_fillPts]
  have e : plainAgreeR i =
      1 - (1 - Real.sqrt ((2 : ℝ) ^ Flock.plainCols i / 2 ^ Flock.plainLen i) - 1 / 200) := by
    unfold plainAgreeR
    ring
  rw [e]
  exact hM

theorem plainDecodeR_complexity (i : Flock.PlainIdx) :
    DecoderComplexity (plainDecodeR i) (plainCountedR i)
      (fun os => (os.length + 2 ^ Flock.plainLen i) * 2 ^ Flock.plainLanes i * 128) flockPolyR := by
  intro os
  have hc := gsDecode_complexity (Flock.flockPlainLevelZero i) rf128 (plainReader i) (plainAgreeR i) 1 128
    (by norm_num) (fillPts (2 ^ Flock.plainLen i) os)
  have hb : gsBound rf128.bound (plainReader i).bound 1 = flockPoly := gsBound_flock
  rw [hb] at hc
  simp only [plainCountedR, plainDecodeR, fillPtsC_fst]
  refine ⟨hc.1, ?_⟩
  have hf := fillPtsC_snd_le (2 ^ Flock.plainLen i) os
  have hg := hc.2
  simp only [length_fillPts] at hg
  simp only [flockPolyR, eval_add, eval_pow, eval_X]
  generalize hQ : (os.length + 2 ^ Flock.plainLen i) * 2 ^ Flock.plainLanes i * 128 = Q
  have hw : 1 ≤ 2 ^ Flock.plainLanes i * 128 := Nat.one_le_iff_ne_zero.2 (by positivity)
  have hnQ : 2 ^ Flock.plainLen i * 2 ^ Flock.plainLanes i * 128 ≤ Q := by
    rw [← hQ, mul_assoc, mul_assoc]
    exact Nat.mul_le_mul_right _ (Nat.le_add_left _ _)
  have hn : 2 ^ Flock.plainLen i ≤ Q := le_trans (by rw [mul_assoc]; exact Nat.le_mul_of_pos_right _ hw) hnQ
  have hL : os.length ≤ Q := by
    rw [← hQ, mul_assoc]
    exact le_trans (Nat.le_add_right _ _) (Nat.le_mul_of_pos_right _ hw)
  have hg' := hg.trans (eval_mono_nat flockPoly hnQ)
  have hf' : 2 ^ Flock.plainLen i * (os.length + 1) ≤ Q ^ 2 + Q := by
    calc 2 ^ Flock.plainLen i * (os.length + 1) ≤ Q * (Q + 1) := Nat.mul_le_mul hn (by omega)
      _ = Q ^ 2 + Q := by ring
  omega

/-- **The plain verifier's radius decoder**: each `m`'s plain radius decoder (`plainDecodeR`) on the opened columns
read at its width. -/
noncomputable def plainDecodeRD (os : List (ℕ × List GF128)) : List Flock.PlainTrace :=
  famDecodeR plainDecodeR os

/-- `plainDecodeRD` with its count. -/
noncomputable def plainCountedRD (os : List (ℕ × List GF128)) : List Flock.PlainTrace × ℕ :=
  famCountedR plainCountedR os

def plainSizeRD (os : List (ℕ × List GF128)) : ℕ :=
  ∑ i : Flock.PlainIdx, (os.length + 2 ^ Flock.plainLen i) * 2 ^ Flock.plainLanes i * 128

theorem listWithin_plainDecodeRD (t : List GF128 → List GF128) (ht : ∀ u ℓ, (t u).getD ℓ 0 = u.getD ℓ 0) :
    Flock.ListWithin Flock.plainLenD (fun M q => t (Flock.plainEncD M q)) Flock.plainThetaD plainDecodeRD :=
  listWithin_famDecodeR (s := fun i => 2 ^ Flock.plainLanes i)
    (T := fun i => Message GF128 (Flock.plainLanes i) (Flock.plainCols i))
    (fun i _ => 2 ^ Flock.plainLen i) (fun i => (Flock.flockPlainLevelZero i).enc)
    (fun i _ => 1 - Real.sqrt ((2 : ℝ) ^ Flock.plainCols i / 2 ^ Flock.plainLen i) - 1 / 200) plainDecodeR
    t ht fun i => by
      rw [Flock.flockPlainLevelZero, Flock.levelZero_enc]
      exact listWithin_plainDecodeR i

theorem plainDecodeRD_complexity :
    DecoderComplexity plainDecodeRD plainCountedRD plainSizeRD (C 14 * (flockPolyR + Polynomial.X)) := by
  have hle : ∀ (i : Flock.PlainIdx) (os : List (ℕ × List GF128)),
      (os.length + 2 ^ Flock.plainLen i) * 2 ^ Flock.plainLanes i * 128 ≤ plainSizeRD os := fun i os =>
    Finset.single_le_sum
      (f := fun i : Flock.PlainIdx => (os.length + 2 ^ Flock.plainLen i) * 2 ^ Flock.plainLanes i * 128)
      (fun _ _ => Nat.zero_le _) (Finset.mem_univ i)
  have := famDecodeR_complexity plainDecodeR plainCountedR
    (fun i os => (os.length + 2 ^ Flock.plainLen i) * 2 ^ Flock.plainLanes i * 128) flockPolyR
    plainDecodeR_complexity plainSizeRD
    (fun i os => by
      rw [IRS.length_colsAt]
      exact hle i os)
    (fun i os => le_trans (calc os.length * 2 ^ Flock.plainLanes i
        _ ≤ (os.length + 2 ^ Flock.plainLen i) * 2 ^ Flock.plainLanes i :=
          Nat.mul_le_mul_right _ (Nat.le_add_right _ _)
        _ ≤ (os.length + 2 ^ Flock.plainLen i) * 2 ^ Flock.plainLanes i * 128 :=
          Nat.le_mul_of_pos_right _ (by norm_num))
      (hle i os))
  rw [Fintype.card_fin] at this
  exact this

theorem plainDecodeRD_length_le (os : List (ℕ × List GF128)) :
    (plainDecodeRD os).length ≤ (C 14 * (flockPolyR + Polynomial.X)).eval (plainSizeRD os) := by
  have hi : ∀ i : Flock.PlainIdx,
      (plainDecodeR i (IRS.colsAt (2 ^ Flock.plainLanes i) os)).length ≤ plainSizeRD os := by
    intro i
    unfold plainDecodeR
    refine (length_gsDecode_le _ _ _ _ _).trans ?_
    rw [length_fillPts, max_eq_right Nat.one_le_two_pow]
    refine le_trans ?_ (Finset.single_le_sum
      (f := fun i : Flock.PlainIdx => (os.length + 2 ^ Flock.plainLen i) * 2 ^ Flock.plainLanes i * 128)
      (fun _ _ => Nat.zero_le _) (Finset.mem_univ i))
    have hw : 1 ≤ 2 ^ Flock.plainLanes i * 128 := Nat.one_le_iff_ne_zero.2 (by positivity)
    calc 2 ^ Flock.plainLen i ≤ os.length + 2 ^ Flock.plainLen i := Nat.le_add_left _ _
      _ ≤ (os.length + 2 ^ Flock.plainLen i) * (2 ^ Flock.plainLanes i * 128) := Nat.le_mul_of_pos_right _ hw
      _ = (os.length + 2 ^ Flock.plainLen i) * 2 ^ Flock.plainLanes i * 128 := by ring
  refine (length_famDecodeR_le plainDecodeR os _ hi).trans ?_
  simp only [Fintype.card_fin, eval_mul, eval_C, eval_add, eval_X]
  exact Nat.mul_le_mul_left 14 (Nat.le_add_left _ _)

theorem sizeTerm_le (L n w : ℕ) (hn : 1 ≤ n) : (L + n) * w * 128 ≤ (0 + n) * w * 128 * (L + 1) := by
  have h : L + n ≤ n * (L + 1) := by
    rw [mul_add_one]
    exact Nat.add_le_add_right (Nat.le_mul_of_pos_left L hn) n
  calc (L + n) * w * 128 ≤ n * (L + 1) * w * 128 := by gcongr
    _ = (0 + n) * w * 128 * (L + 1) := by ring

theorem plainSizeRD_le (os : List (ℕ × List GF128)) : plainSizeRD os ≤ plainSizeRD [] * (os.length + 1) := by
  unfold plainSizeRD
  rw [Finset.sum_mul]
  exact Finset.sum_le_sum fun i _ => sizeTerm_le _ _ _ Nat.one_le_two_pow

theorem zkSizeRD_le (os : List (ℕ × List GF128)) : zkSizeRD os ≤ zkSizeRD [] * (os.length + 1) := by
  unfold zkSizeRD
  rw [Finset.sum_mul]
  exact Finset.sum_le_sum fun i _ => sizeTerm_le _ _ _ Nat.one_le_two_pow

theorem plainCountedRD_fst (os : List (ℕ × List GF128)) : (plainCountedRD os).1 = plainDecodeRD os :=
  (plainDecodeRD_complexity os).1

theorem zkCountedRD_fst (os : List (ℕ × List GF128)) : (zkCountedRD os).1 = zkDecodeRD os :=
  (zkDecodeRD_complexity os).1

theorem fillPtsC_nil_snd {Col : Type} [Zero Col] (n : ℕ) : (fillPtsC n ([] : List (ℕ × Col))).2 = n := by
  simp [fillPtsC, lookupC]

theorem plainCountedRD_nil_pos : 0 < (plainCountedRD []).2 := by
  have h0 : 0 < (plainCountedR 0 (IRS.colsAt (2 ^ Flock.plainLanes 0) [])).2 := by
    have : (fillPtsC (2 ^ Flock.plainLen 0) (IRS.colsAt (2 ^ Flock.plainLanes 0) ([] : List (ℕ × List GF128)))).2 =
        2 ^ Flock.plainLen 0 := fillPtsC_nil_snd _
    simp only [plainCountedR, this]
    exact Nat.add_pos_left (Nat.two_pow_pos _) _
  unfold plainCountedRD famCountedR
  exact lt_of_lt_of_le (lt_of_lt_of_le h0 (Nat.le_add_left _ _))
    (Finset.single_le_sum (f := fun i : Flock.PlainIdx =>
      ([] : List (ℕ × List GF128)).length * 2 ^ Flock.plainLanes i +
        (plainCountedR i (IRS.colsAt (2 ^ Flock.plainLanes i) [])).2) (fun _ _ => Nat.zero_le _) (Finset.mem_univ 0))

theorem zkCountedRD_nil_pos : 0 < (zkCountedRD []).2 := by
  have h0 : 0 < (zkCountedR 0 (IRS.colsAt (Flock.zkWidth 0) [])).2 := by
    have : (fillPtsC (2 ^ Flock.zkLen 0) (IRS.colsAt (Flock.zkWidth 0) ([] : List (ℕ × List GF128)))).2 =
        2 ^ Flock.zkLen 0 := fillPtsC_nil_snd _
    simp only [zkCountedR, this]
    exact Nat.add_pos_left (Nat.two_pow_pos _) _
  unfold zkCountedRD famCountedR
  exact lt_of_lt_of_le (lt_of_lt_of_le h0 (Nat.le_add_left _ _))
    (Finset.single_le_sum (f := fun i : Flock.ZkIdx =>
      ([] : List (ℕ × List GF128)).length * Flock.zkWidth i +
        (zkCountedR i (IRS.colsAt (Flock.zkWidth i) [])).2) (fun _ _ => Nat.zero_le _) (Finset.mem_univ 0))

end Claims.GS
