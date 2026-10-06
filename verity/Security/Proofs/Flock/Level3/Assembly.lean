/-
Level 3, L3-A (PROTOCOL.md §17.4 stage 1d): the assembly. First the equality tensor: the executable's `eqTable r` holds,
at index `b`, `eq(r, b) = Π_j (r_j if bit j of b, else 1 + r_j)` (the soundness model's `eqAt`), over the field GF128.
-/
import Proofs.Flock.Level3.GF128Field
import Flock.Piop
import Mathlib.Algebra.BigOperators.Fin

open Flock

namespace FlockLevel3

open GF128

/-- The factor of coordinate `x` for index bit `bit`. -/
noncomputable def eqFactor (bit : Bool) (x : F128) : GF128 := if bit then of x else 1 + of x

theorem eqStep_size (t : Array F128) (x : F128) : (eqStep t x).size = 2 * t.size := by
  simp [eqStep]; omega

theorem eqStep_get (t : Array F128) (x : F128) (i : ℕ) (hi : i < (eqStep t x).size) (ht : 0 < t.size) :
    of ((eqStep t x)[i]) = of (t[i % t.size]'(Nat.mod_lt _ ht)) * eqFactor ((i / t.size) % 2 == 1) x := by
  have hi' : i < 2 * t.size := by rw [eqStep_size] at hi; exact hi
  have key : (eqStep t x)[i] = (t.map (· * (x + 1)) ++ t.map (· * x))[i]'(by simp; omega) := rfl
  rw [key]
  unfold eqFactor
  simp only [Array.getElem_append, Array.size_map, Array.getElem_map]
  by_cases h : i < t.size
  · have hm : i % t.size = i := Nat.mod_eq_of_lt h
    have hd : i / t.size = 0 := Nat.div_eq_of_lt h
    simp only [h, dite_true, hd, Nat.zero_mod]
    rw [← mul_eq, ← add_eq, ← one_eq]
    simp only [hm]
    simp [add_comm]
  · have hm : i % t.size = i - t.size := by
      rw [Nat.mod_eq_sub_mod (by omega), Nat.mod_eq_of_lt (by omega)]
    have hd : i / t.size = 1 := by
      exact Nat.div_eq_of_lt_le (by omega) (by omega)
    simp only [h, dite_false, hd]
    rw [← mul_eq]
    simp only [hm]
    simp

theorem foldl_eqStep_size (l : List F128) : ∀ t : Array F128, (l.foldl eqStep t).size = t.size * 2 ^ l.length := by
  induction l with
  | nil => intro t; simp
  | cons x l ih => intro t; rw [List.foldl_cons, ih, eqStep_size, List.length_cons, pow_succ]; ring

theorem foldl_eqStep_get (l : List F128) : ∀ (t : Array F128) (ht : 0 < t.size) (i : ℕ) (hi : i < (l.foldl eqStep t).size),
    of ((l.foldl eqStep t)[i]) =
      of (t[i % t.size]'(Nat.mod_lt _ ht)) * ∏ j : Fin l.length, eqFactor ((i / t.size).testBit j) l[j] := by
  induction l with
  | nil =>
    intro t ht i hi
    simp only [List.foldl_nil] at hi ⊢
    simp [Nat.mod_eq_of_lt hi]
  | cons x l ih =>
    intro t ht i hi
    have ht2 : 0 < (eqStep t x).size := by rw [eqStep_size]; omega
    simp only [List.foldl_cons] at hi ⊢
    rw [ih (eqStep t x) ht2 i hi]
    have him : i % (eqStep t x).size < (eqStep t x).size := Nat.mod_lt _ ht2
    show _ = of (t[i % t.size]'(Nat.mod_lt _ ht)) * ∏ j : Fin (l.length + 1), eqFactor ((i / t.size).testBit j) (x :: l)[j]
    rw [eqStep_get t x _ him ht, Fin.prod_univ_succ]
    have e1 : i % (eqStep t x).size % t.size = i % t.size := by
      rw [eqStep_size, Nat.mod_mul_left_mod]
    have e2 : i % (eqStep t x).size / t.size % 2 = i / t.size % 2 := by
      rw [eqStep_size, mul_comm, Nat.mod_mul_right_div_self, Nat.mod_mod]
    have e3 : ∀ j : ℕ, (i / (eqStep t x).size).testBit j = (i / t.size).testBit (j + 1) := by
      intro j
      rw [eqStep_size, mul_comm, ← Nat.div_div_eq_div_mul, Nat.testBit_div_two]
    simp only [e1, e2, e3, mul_assoc]
    congr 1
    congr 1
    simp [Nat.testBit_zero, eqFactor]

/-- **The equality tensor.** `eqTable r` has `2^|r|` entries, and entry `b` is `eq(r, b)`. -/
theorem eqTable_size (r : Array F128) : (eqTable r).size = 2 ^ r.size := by
  rw [eqTable, ← Array.foldl_toList, foldl_eqStep_size]; simp

theorem eqTable_get (r : Array F128) (b : ℕ) (hb : b < (eqTable r).size) :
    of ((eqTable r)[b]) = ∏ j : Fin r.size, eqFactor (b.testBit j) r[j] := by
  have h := foldl_eqStep_get r.toList #[1] (by simp) b (by rwa [Array.foldl_toList])
  simp only [Array.foldl_toList] at h
  have e : of (#[(1 : F128)][b % #[(1 : F128)].size]'(Nat.mod_lt _ (by simp))) = 1 := by
    rw [one_eq]; congr 1; simp
  simp only [eqTable]
  rw [h, e, one_mul, show (#[(1 : F128)] : Array F128).size = 1 from rfl, Nat.div_one]
  exact Fintype.prod_equiv (finCongr (by simp)) _ _ (fun j => by simp)

end FlockLevel3
