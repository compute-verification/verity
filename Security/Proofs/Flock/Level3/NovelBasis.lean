/-
`Arith.Correct.xhat_poly` (§13.1, §13.4): the novel basis. `s_0 = X`, `s_{j+1} = s_j² + s_j(β_j)·s_j` over `β_j = x^j`
(monic, degree `2^j`); `Ŵ_i = s_i · s_i(β_i)⁻¹`; `X̂_c = Π_{i ∈ bits c} Ŵ_i` is a polynomial of degree exactly `c`, so the
`X̂_c`, `c < 2^L`, are linearly independent. The executable's `sIter` and `svTable` compute these values, and the kernel
checks `s_i(β_i) ≠ 0` for `i < 40` on `svTable 40`.
-/
import Proofs.Flock.Level3.Nodes
import Flock.Ligerito
import Mathlib.Algebra.Polynomial.Sequence
import Proofs.Flock.Level3.NovelBasis.Steps0
import Proofs.Flock.Level3.NovelBasis.Steps14
import Proofs.Flock.Level3.NovelBasis.Steps20
import Proofs.Flock.Level3.NovelBasis.Steps24
import Proofs.Flock.Level3.NovelBasis.Steps28
import Proofs.Flock.Level3.NovelBasis.Steps31
import Proofs.Flock.Level3.NovelBasis.Steps34
import Proofs.Flock.Level3.NovelBasis.Steps37

open Polynomial Flock

set_option maxRecDepth 20000

namespace FlockLevel3

noncomputable def beta (j : ℕ) : GF128 := GF128.of (F128.monomial j)

noncomputable def sPoly : ℕ → GF128[X]
  | 0 => X
  | j + 1 => sPoly j ^ 2 + C ((sPoly j).eval (beta j)) * sPoly j

/-- `Ŵ_i(ω) = s_i(ω) · s_i(β_i)⁻¹` (the soundness model's `What`). -/
noncomputable def What (i : ℕ) (ω : GF128) : GF128 := (sPoly i).eval ω * ((sPoly i).eval (beta i))⁻¹

/-- `X̂_c(ω) = Π_{i ∈ bits(c)} Ŵ_i(ω)` (the model's `Arith.Xhat`). -/
noncomputable def Xhat (c L : ℕ) (ω : GF128) : GF128 := ∏ i : Fin L, if c.testBit i then What i ω else 1

theorem sPoly_monic_natDegree : ∀ j, (sPoly j).Monic ∧ (sPoly j).natDegree = 2 ^ j
  | 0 => ⟨monic_X, by simp [sPoly]⟩
  | j + 1 => by
    obtain ⟨hm, hd⟩ := sPoly_monic_natDegree j
    have hsq : (sPoly j ^ 2).Monic := hm.pow 2
    have hsqd : (sPoly j ^ 2).natDegree = 2 ^ (j + 1) := by rw [hm.natDegree_pow, hd, pow_succ, mul_comm]
    have hlt : (C ((sPoly j).eval (beta j)) * sPoly j).degree < (sPoly j ^ 2).degree := by
      apply degree_lt_degree
      calc (C ((sPoly j).eval (beta j)) * sPoly j).natDegree ≤ (sPoly j).natDegree := natDegree_C_mul_le _ _
        _ = 2 ^ j := hd
        _ < 2 ^ (j + 1) := Nat.pow_lt_pow_right (by norm_num) (by omega)
        _ = (sPoly j ^ 2).natDegree := hsqd.symm
    refine ⟨hsq.add_of_left hlt, ?_⟩
    show (sPoly j ^ 2 + C ((sPoly j).eval (beta j)) * sPoly j).natDegree = 2 ^ (j + 1)
    rw [natDegree_add_eq_left_of_degree_lt hlt, hsqd]

/-! ### The executable computes these values -/

theorem sIter_eval (sv : Array F128) (k : ℕ) (hsv : ∀ j < k, GF128.of sv[j]! = (sPoly j).eval (beta j)) (x : F128) :
    GF128.of (sIter sv k x) = (sPoly k).eval (GF128.of x) := by
  induction k with
  | zero => simp [sIter, sPoly]
  | succ k ih =>
    simp only [sIter]
    rw [of_add, of_mul, of_mul, ih (fun j hj => hsv j (by omega)), hsv k (by omega)]
    simp only [sPoly, eval_add, eval_pow, eval_mul, eval_C]
    ring

theorem svTable_size : ∀ n, (svTable n).size = n
  | 0 => rfl
  | n + 1 => by simp [svTable, svTable_size n]

theorem svTable_push_get (n j : ℕ) (hj : j < n + 1) :
    (svTable (n + 1))[j]! = if j < n then (svTable n)[j]! else sIter (svTable n) n (F128.monomial n) := by
  simp only [svTable]
  rw [getElem!_pos _ j (by simp [svTable_size]; omega), Array.getElem_push]
  split_ifs with h1 h2 h2
  · rw [getElem!_pos _ j h1]
  · simp [svTable_size] at h1; omega
  · simp [svTable_size] at h1; omega
  · rfl

theorem sIter_congr (sv sv' : Array F128) (k : ℕ) (h : ∀ j < k, sv[j]! = sv'[j]!) (x : F128) :
    sIter sv k x = sIter sv' k x := by
  induction k with
  | zero => rfl
  | succ k ih => simp only [sIter, ih (fun j hj => h j (by omega)), h k (by omega)]

theorem svTable_eval : ∀ n, ∀ j < n, GF128.of (svTable n)[j]! = (sPoly j).eval (beta j)
  | 0 => fun j hj => absurd hj (Nat.not_lt_zero _)
  | n + 1 => fun j hj => by
    rw [svTable_push_get n j hj]
    split_ifs with h
    · exact svTable_eval n j h
    · have hj' : j = n := by omega
      subst hj'
      rw [sIter_eval _ j (svTable_eval j)]; rfl

theorem svC_step : ∀ i : Fin 40, sIter svC i (F128.monomial i) = svC[i.val]! := fun i =>
  match i with
  | ⟨0, _⟩ => svC_step_0
  | ⟨1, _⟩ => svC_step_1
  | ⟨2, _⟩ => svC_step_2
  | ⟨3, _⟩ => svC_step_3
  | ⟨4, _⟩ => svC_step_4
  | ⟨5, _⟩ => svC_step_5
  | ⟨6, _⟩ => svC_step_6
  | ⟨7, _⟩ => svC_step_7
  | ⟨8, _⟩ => svC_step_8
  | ⟨9, _⟩ => svC_step_9
  | ⟨10, _⟩ => svC_step_10
  | ⟨11, _⟩ => svC_step_11
  | ⟨12, _⟩ => svC_step_12
  | ⟨13, _⟩ => svC_step_13
  | ⟨14, _⟩ => svC_step_14
  | ⟨15, _⟩ => svC_step_15
  | ⟨16, _⟩ => svC_step_16
  | ⟨17, _⟩ => svC_step_17
  | ⟨18, _⟩ => svC_step_18
  | ⟨19, _⟩ => svC_step_19
  | ⟨20, _⟩ => svC_step_20
  | ⟨21, _⟩ => svC_step_21
  | ⟨22, _⟩ => svC_step_22
  | ⟨23, _⟩ => svC_step_23
  | ⟨24, _⟩ => svC_step_24
  | ⟨25, _⟩ => svC_step_25
  | ⟨26, _⟩ => svC_step_26
  | ⟨27, _⟩ => svC_step_27
  | ⟨28, _⟩ => svC_step_28
  | ⟨29, _⟩ => svC_step_29
  | ⟨30, _⟩ => svC_step_30
  | ⟨31, _⟩ => svC_step_31
  | ⟨32, _⟩ => svC_step_32
  | ⟨33, _⟩ => svC_step_33
  | ⟨34, _⟩ => svC_step_34
  | ⟨35, _⟩ => svC_step_35
  | ⟨36, _⟩ => svC_step_36
  | ⟨37, _⟩ => svC_step_37
  | ⟨38, _⟩ => svC_step_38
  | ⟨39, _⟩ => svC_step_39
  | ⟨n + 40, h⟩ => absurd h (by omega)

theorem svTable_eq_svC : ∀ n ≤ 40, ∀ j < n, (svTable n)[j]! = svC[j]!
  | 0, _ => fun j hj => absurd hj (Nat.not_lt_zero _)
  | n + 1, hn => fun j hj => by
    rw [svTable_push_get n j hj]
    split_ifs with h
    · exact svTable_eq_svC n (by omega) j h
    · have hj' : j = n := by omega
      subst hj'
      rw [sIter_congr _ svC j (svTable_eq_svC j (by omega))]
      exact svC_step ⟨j, by omega⟩

theorem svC_ne_zero : svC.all (fun x => decide (x ≠ F128.zero)) = true := by decide +kernel

theorem svC_size : svC.size = 40 := rfl

theorem sPoly_beta_ne_zero (i : ℕ) (hi : i < 40) : (sPoly i).eval (beta i) ≠ 0 := by
  rw [← svTable_eval 40 i hi, svTable_eq_svC 40 le_rfl i hi, GF128.zero_eq]
  have hi' : i < svC.size := by rw [svC_size]; exact hi
  have h := Array.all_eq_true.mp svC_ne_zero i hi'
  rw [← getElem!_pos svC i hi'] at h
  intro h0; exact of_decide_eq_true h h0

/-! ### `X̂_c` is a polynomial of degree `c` -/

noncomputable def WPoly (i : ℕ) : GF128[X] := C ((sPoly i).eval (beta i))⁻¹ * sPoly i

noncomputable def Pc (L c : ℕ) : GF128[X] := ∏ i : Fin L, if c.testBit i then WPoly i else 1

theorem eval_Pc (L c : ℕ) (ω : GF128) : (Pc L c).eval ω = Xhat c L ω := by
  rw [Pc, eval_prod]
  refine Finset.prod_congr rfl fun i _ => ?_
  split_ifs <;> simp [WPoly, What, mul_comm]

theorem WPoly_ne_zero (i : ℕ) (hi : i < 40) : WPoly i ≠ 0 :=
  mul_ne_zero (C_ne_zero.mpr (inv_ne_zero (sPoly_beta_ne_zero i hi))) (sPoly_monic_natDegree i).1.ne_zero

theorem WPoly_natDegree (i : ℕ) (hi : i < 40) : (WPoly i).natDegree = 2 ^ i := by
  rw [WPoly, natDegree_C_mul (inv_ne_zero (sPoly_beta_ne_zero i hi)), (sPoly_monic_natDegree i).2]

theorem sum_bits : ∀ (L c : ℕ), c < 2 ^ L → ∑ i : Fin L, (if c.testBit i then 2 ^ (i : ℕ) else 0) = c
  | 0, c, hc => by simp at hc; subst hc; simp
  | L + 1, c, hc => by
    rw [Fin.sum_univ_castSucc]
    have hlow : ∑ i : Fin L, (if c.testBit (Fin.castSucc i) then 2 ^ ((Fin.castSucc i : Fin (L + 1)) : ℕ) else 0) =
        c % 2 ^ L := by
      rw [← sum_bits L (c % 2 ^ L) (Nat.mod_lt _ (Nat.two_pow_pos L))]
      refine Finset.sum_congr rfl fun i _ => ?_
      simp [Nat.testBit_mod_two_pow, i.isLt]
    rw [hlow]
    have hq : c / 2 ^ L < 2 := by
      rw [Nat.div_lt_iff_lt_mul (Nat.two_pow_pos L)]; rw [pow_succ, mul_comm] at hc; exact hc
    have htb : c.testBit L = decide (c / 2 ^ L = 1) := by
      rw [Nat.testBit_eq_decide_div_mod_eq]
      rcases Nat.lt_succ_iff_lt_or_eq.mp hq with h | h
      · have h0 : c / 2 ^ L = 0 := Nat.lt_one_iff.mp h
        simp [h0]
      · simp [h]
    simp only [Fin.val_last, htb]
    have hmd := Nat.mod_add_div c (2 ^ L)
    split_ifs with h
    · have h1 : c / 2 ^ L = 1 := of_decide_eq_true h
      rw [h1, mul_one] at hmd; exact hmd
    · have h1 : c / 2 ^ L ≠ 1 := fun h' => h (decide_eq_true h')
      have h0 : c / 2 ^ L = 0 := by
        rcases Nat.lt_succ_iff_lt_or_eq.mp hq with h' | h'
        · exact Nat.lt_one_iff.mp h'
        · exact absurd h' h1
      rw [h0, mul_zero, add_zero] at hmd; rw [add_zero]; exact hmd

theorem Pc_natDegree (L : ℕ) (hL : L ≤ 40) (c : ℕ) (hc : c < 2 ^ L) : (Pc L c).natDegree = c := by
  have hne : ∀ i ∈ (Finset.univ : Finset (Fin L)), (if c.testBit i then WPoly i else 1) ≠ 0 := by
    intro i _; split_ifs
    · exact WPoly_ne_zero i (by omega)
    · exact one_ne_zero
  rw [Pc, natDegree_prod _ _ hne]
  conv_rhs => rw [← sum_bits L c hc]
  refine Finset.sum_congr rfl fun i _ => ?_
  split_ifs
  · exact WPoly_natDegree i (by omega)
  · exact natDegree_one

theorem Pc_ne_zero (L : ℕ) (hL : L ≤ 40) (c : ℕ) : Pc L c ≠ 0 :=
  Finset.prod_ne_zero_iff.mpr fun i _ => by split_ifs; exact WPoly_ne_zero i (by omega); exact one_ne_zero

/-- The family `P_c` for `c < 2^L`, padded with `X^c` past it, as a `Polynomial.Sequence`. -/
noncomputable def seqP (L : ℕ) (hL : L ≤ 40) : Polynomial.Sequence GF128 where
  elems' c := if c < 2 ^ L then Pc L c else X ^ c
  degree_eq' c := by
    split_ifs with hc
    · rw [degree_eq_natDegree (Pc_ne_zero L hL c), Pc_natDegree L hL c hc]
    · exact degree_X_pow c

/-- **`Arith.Correct.xhat_poly`**, for `L ≤ 40`. -/
theorem xhat_poly : ∀ L ≤ 40, ∃ P : Fin (2 ^ L) → GF128[X], (∀ c, (P c).natDegree < 2 ^ L) ∧
    LinearIndependent GF128 P ∧ ∀ (c : Fin (2 ^ L)) ω, Xhat c.val L ω = (P c).eval ω := by
  intro L hL
  refine ⟨fun c => Pc L c.val, fun c => by rw [Pc_natDegree L hL c.val c.isLt]; exact c.isLt, ?_,
    fun c ω => (eval_Pc L c.val ω).symm⟩
  have hli := (seqP L hL).linearIndependent
  have : (fun c : Fin (2 ^ L) => Pc L c.val) = (seqP L hL) ∘ Fin.val := by
    funext c; show Pc L c.val = if c.val < 2 ^ L then Pc L c.val else X ^ c.val; rw [if_pos c.isLt]
  rw [this]
  exact hli.comp _ Fin.val_injective

end FlockLevel3
