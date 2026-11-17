module

public import Proofs.Flock.VBridge.RecOpen
public import Flock.Verify
import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.MeasureTheory.Integral.Bochner.Basic
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
import Mathlib.Tactic.ReduceModChar
import Proofs.Flock.Soundness.Refine.Arith
import VCVio.Prelude.Core

/-! The words the auditor computes for V*, as Lean functions of the inner statement and the rep's coins: the shape, the openings' coefficients and their running sums. -/

@[expose] public section

namespace FlockVBridge

open Flock FlockSoundness.Refine

local notation "𝔽" => FlockLevel3.GF128

def repShape (st : Setup) (sch : Schedule) (extras : Array Claim) : Shape :=
  { m := st.m, kLog := st.kLog, claims := 2 + extras.size, sched := sch, hashLen := st.merkle.digestLen,
    saltLen := if st.merkle.saltBytes = 0 then none else some st.merkle.saltBytes }

def coefAt (kap : Array (Array F128)) (LANES : ℕ) (e : F128) : Array F128 :=
  Array.ofFn (n := LANES) (fun l => (kap[0]!)[l.1]! * e) ++ Array.ofFn (n := LANES) (fun l => (kap[1]!)[l.1]! * e)

theorem coefAt_get (kap : Array (Array F128)) (LANES : ℕ) (e : F128) {k l : ℕ} (hk : k < 2) (hl : l < LANES) :
    (coefAt kap LANES e)[k * LANES + l]! = (kap[k]!)[l]! * e := by
  rcases (by omega : k = 0 ∨ k = 1) with rfl | rfl
  · simp [coefAt, getElem!_def, Array.getElem?_append, hl]
  · simp [coefAt, getElem!_def, hl]

theorem fOf_foldl_add (g : ℕ → F128) : ∀ (n : ℕ) (x : F128),
    fOf (((List.range n).map g).foldl (· + ·) x) = fOf x + ∑ l ∈ Finset.range n, fOf (g l)
  | 0, x => by simp
  | n + 1, x => by
    rw [List.range_succ, List.map_append, List.foldl_append, List.map_cons, List.map_nil, List.foldl_cons,
      List.foldl_nil, fOf_add, fOf_foldl_add g n x, Finset.sum_range_succ, add_assoc]

theorem fOf_consTerm_coefAt (LANES : ℕ) (kap : Array (Array F128)) (e : F128) (row a : Array F128) {k : ℕ}
    (hk : k < 2) : fOf (consTerm LANES row a (coefAt kap LANES e) k) =
      fOf a[k]! + ∑ l ∈ Finset.range LANES, fOf (kap[k]!)[l]! * (fOf e * fOf row[l]!) := by
  rw [consTerm, fOf_foldl_add]
  refine congrArg (fOf a[k]! + ·) (Finset.sum_congr rfl fun l hl => ?_)
  rw [Finset.mem_range] at hl
  rw [coefAt_get kap LANES e hk hl, fOf_mul, fOf_mul, mul_assoc]

theorem sum_map_range (f : ℕ → 𝔽) : ∀ q, ((List.range q).map f).sum = ∑ j ∈ Finset.range q, f j
  | 0 => by simp
  | q + 1 => by
    rw [List.range_succ, List.map_append, List.sum_append, sum_map_range f q, Finset.sum_range_succ]
    simp

def chainAcc (LANES : ℕ) (kap : Array (Array F128)) : List (F128 × Array F128) → Array F128 → Array F128
  | [], a => a
  | o :: os, a => chainAcc LANES kap os
      #[consTerm LANES o.2 a (coefAt kap LANES o.1) 0, consTerm LANES o.2 a (coefAt kap LANES o.1) 1]

theorem chainAcc_sum (LANES : ℕ) (kap : Array (Array F128)) :
    ∀ (os : List (F128 × Array F128)) (a : Array F128) {k : ℕ}, k < 2 →
      fOf (chainAcc LANES kap os a)[k]! =
        fOf a[k]! + ∑ l ∈ Finset.range LANES, fOf (kap[k]!)[l]! * (os.map fun o => fOf o.1 * fOf o.2[l]!).sum
  | [], a, k, _ => by simp [chainAcc]
  | o :: os, a, k, hk => by
    rw [chainAcc, chainAcc_sum LANES kap os _ hk]
    have h : (#[consTerm LANES o.2 a (coefAt kap LANES o.1) 0, consTerm LANES o.2 a (coefAt kap LANES o.1) 1] :
        Array F128)[k]! = consTerm LANES o.2 a (coefAt kap LANES o.1) k := by
      rcases (by omega : k = 0 ∨ k = 1) with rfl | rfl <;> rfl
    rw [h, fOf_consTerm_coefAt LANES kap o.1 o.2 a hk, add_assoc, ← Finset.sum_add_distrib]
    refine congrArg (fOf a[k]! + ·) (Finset.sum_congr rfl fun l _ => ?_)
    rw [List.map_cons, List.sum_cons, mul_add]

theorem chainAcc_level (LANES : ℕ) (kap : Array (Array F128)) (rows : Array (Array F128)) (q : ℕ)
    (alpha a : Array F128) {k : ℕ} (hk : k < 2) :
    fOf (chainAcc LANES kap ((List.range q).map fun j => ((eqTable alpha)[j]!, rows[j]!)) a)[k]! + fOf a[k]! =
      ∑ l ∈ Finset.range LANES, fOf (kap[k]!)[l]! * ∑ j ∈ Finset.range q, fOf (eqTable alpha)[j]! * fOf (rows[j]!)[l]! := by
  rw [chainAcc_sum LANES kap _ a hk, add_comm (fOf a[k]!) _, add_assoc, CharTwo.add_self_eq_zero, add_zero]
  refine Finset.sum_congr rfl fun l _ => ?_
  rw [List.map_map, sum_map_range]
  rfl

end FlockVBridge
