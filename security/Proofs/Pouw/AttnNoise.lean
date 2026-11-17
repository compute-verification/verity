module

public import Proofs.Pouw.RelativeSecurity
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
import Mathlib.Topology.Sheaves.Init

/-! Attention's noise: a committed prefix word's floor holds at every operand value, and one-sided noise has
none. -/

@[expose] public section

namespace Pouw.RelativeSecurity

open Finset

section AttnNoise

variable {R : Type} [CommRing R] {n L np B n' : ℕ}

theorem dPrefix_eq_sum (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) (w : Fin np → R)
    (i : Fin n) (l : Fin n') (t : ℕ) :
    dPrefix pc vq w i l t = fun z => ∑ t' : Fin B, (if t'.val ≤ t then 1 else 0) * dIncr pc vq w i l t' z := by
  funext z
  simp only [dPrefix, ite_mul, one_mul, zero_mul]

theorem pairDiff_dPrefix_operands (pc pc' : Fin n → Fin B → Fin L → R) (vq vq' : Fin B → Fin L → Fin n' → R)
    (w : Fin np → R) (i : Fin n) (l : Fin n') (t : ℕ) {u v : DBits n L np B n'} (huv : u ≠ v) :
    pairDiff (dPrefix pc vq w i l t) u v = pairDiff (dPrefix pc' vq' w i l t) u v := by
  rw [dPrefix_eq_sum, dPrefix_eq_sum, pairDiff_sum, pairDiff_sum]
  refine Finset.sum_congr rfl fun t' _ => ?_
  rw [pairDiff_mul_left, pairDiff_mul_left, pairDiff_dIncr _ _ _ _ _ _ huv, pairDiff_dIncr _ _ _ _ _ _ huv]
  rfl

def dClean (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) (i : Fin n) (l : Fin n') (t : ℕ) :
    BitAff (DBits n L np B n') R :=
  ⟨∑ t' : Fin B, if t'.val ≤ t then ∑ j, pc i t' j * vq t' j l else 0,
    Sum.elim (fun _ => 0) fun b => if b.2.2 = l then (if b.1.val ≤ t then pc i b.1 b.2.1 else 0) else 0⟩

theorem dClean_eval (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) (i : Fin n) (l : Fin n')
    (t : ℕ) (z : DBits n L np B n' → R) :
    (dClean (np := np) pc vq i l t).eval z =
      ∑ t' : Fin B, if t'.val ≤ t then ∑ j, pc i t' j * (dX vq t' j l).eval z else 0 := by
  simp only [dX_eval]
  rw [BitAff.eval]
  simp only [dClean, Fintype.sum_sum_type, Sum.elim_inl, Sum.elim_inr, zero_mul, Finset.sum_const_zero, zero_add,
    Fintype.sum_prod_type, ite_mul, Finset.sum_ite_eq', Finset.mem_univ, ite_true]
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun t' _ => ?_
  by_cases h : t'.val ≤ t <;> simp [h, mul_add, Finset.sum_add_distrib]

theorem dIncr_split (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) (w : Fin np → R)
    (i : Fin n) (l : Fin n') (t : Fin B) (z : DBits n L np B n' → R) :
    dIncr pc vq w i l t z = dIncr 0 vq w i l t z + ∑ j, pc i t j * (dX vq t j l).eval z := by
  unfold dIncr
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [dP_eval pc, dP_eval 0]
  simp only [Pi.zero_apply]
  ring

theorem dPrefix_split (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) (w : Fin np → R)
    (i : Fin n) (l : Fin n') (t : ℕ) (z : DBits n L np B n' → R) :
    dPrefix pc vq w i l t z = dPrefix 0 vq w i l t z + (dClean pc vq i l t).eval z := by
  rw [dClean_eval]
  simp only [dPrefix, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun t' _ => ?_
  split_ifs
  · exact dIncr_split pc vq w i l t' z
  · simp

theorem crossRankLe_dClean (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) (i : Fin n)
    (l : Fin n') (t : ℕ) : CrossRankLe (fun z => (dClean (np := np) pc vq i l t).eval z) 0 :=
  ⟨Fin.elim0, Fin.elim0, fun _ _ => by simpa using pairDiff_aff _ Sum.inl_ne_inr⟩

theorem pairDiff_dPrefix_noNoise (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R)
    (i : Fin n) (l : Fin n') (t : ℕ) {u v : DBits n L np B n'} (huv : u ≠ v) :
    pairDiff (dPrefix pc vq 0 i l t) u v = 0 := by
  rw [dPrefix_eq_sum, pairDiff_sum]
  refine Finset.sum_eq_zero fun t' _ => ?_
  rw [pairDiff_mul_left, pairDiff_dIncr _ _ _ _ _ _ huv]
  cases u <;> cases v <;> simp [dP]

theorem not_chainBound_noNoise (hL : 0 < L) (hn : 0 < n) (hn' : 0 < n') (hB : 0 < B)
    (Ok : ((DBits n L np B n' → R) → R) → Prop) (c : ℕ) : ¬ ChainBound R n L np B n' 0 Ok c := by
  intro h
  have := h Empty Unit ∅ (fun k => k.elim) (by simp) 0 0 ⟨0, hn⟩ (fun _ => (⟨0, hn'⟩, ⟨0, hB⟩))
    (fun _ _ _ => rfl) (fun _ k => k.elim) (fun _ => ⟨0, fun _ => 0⟩) (fun _ U _ => by
      simp [dPrefix, dIncr, BitAff.eval, dP])
  simp at this
  omega

end AttnNoise

end Pouw.RelativeSecurity
