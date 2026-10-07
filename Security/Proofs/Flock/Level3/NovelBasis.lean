/-
`Arith.Correct.xhat_poly` (§13.1, §13.4): the novel basis. `s_0 = X`, `s_{j+1} = s_j² + s_j(β_j)·s_j` over `β_j = x^j`
(monic, degree `2^j`); `Ŵ_i = s_i · s_i(β_i)⁻¹`; `X̂_c = Π_{i ∈ bits c} Ŵ_i` is a polynomial of degree exactly `c`, so the
`X̂_c`, `c < 2^L`, are linearly independent. The executable's `sIter` and `svTable` compute these values, and the kernel
checks `s_i(β_i) ≠ 0` for `i < 40` on `svTable 40`.
-/
import Proofs.Flock.Level3.Nodes
import Flock.Ligerito
import Mathlib.Algebra.Polynomial.Sequence

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

/-- `s_i(β_i)` for `i < 40`, computed natively; each is checked by the kernel on its own (`svC_step_i`). -/
def svC : Array F128 := #[⟨0x1, 0x0⟩,
  ⟨0x6, 0x0⟩,
  ⟨0x168, 0x0⟩,
  ⟨0x1166880, 0x0⟩,
  ⟨0x116166868808000, 0x1⟩,
  ⟨0x6880808780878c62, 0x116166816686880⟩,
  ⟨0x977dcfdeca5763a3, 0xe4e99706977dcfa5⟩,
  ⟨0xb9db1ae235f97b78, 0xd738e365b46abbf3⟩,
  ⟨0x30f75909af71f18, 0xdb3f8356fe4c8aae⟩,
  ⟨0x7206f70a338f4695, 0xd8e6a10ef60e4734⟩,
  ⟨0x51dd7adac28a6297, 0x8ee9519cd23466ce⟩,
  ⟨0x5ff07b8f5c1f9d6a, 0xfbd61fe266c82bf2⟩,
  ⟨0x88c3d85b6c32f957, 0x8f4a1c96f72deb13⟩,
  ⟨0xa79d9858273d54c, 0x83ad3f8aaaa6e6a2⟩,
  ⟨0x48d8ee13015f698e, 0x69a798e77cf5b06e⟩,
  ⟨0xe06ba4468e8c21cc, 0x8ee54c8e8ae297ba⟩,
  ⟨0xaaa2b91320af4f9a, 0x267da383a0b4a6ec⟩,
  ⟨0x2d24309579f502ec, 0x30f556da177d94c8⟩,
  ⟨0x517cd50cfd167ed4, 0x4a0793046f3e66c0⟩,
  ⟨0xfb14b88ada747186, 0x1399f4f9afb56ae0⟩,
  ⟨0x781b54b7ed1db6cb, 0x836a410cea8384d2⟩,
  ⟨0x7458fdfb343d809f, 0xb1c201bdfda6af9d⟩,
  ⟨0xcde3753a43aa9865, 0x6379afc27b1cb40⟩,
  ⟨0xd119d7ce1e4de8bd, 0xf8bd0542e9d1c09b⟩,
  ⟨0x8447841ac1efb10c, 0xe19b1423499f5bdd⟩,
  ⟨0xd88934c2e6fe7c14, 0x99baf0a32c6f9cb9⟩,
  ⟨0xdcb4f9b26cd865b8, 0xc8a3ee15787627e0⟩,
  ⟨0xb64bd5f8acb8dbc1, 0xde2a33f838331e7c⟩,
  ⟨0xeb7a1fbbf20aaa6f, 0x916f7d658430ba75⟩,
  ⟨0x54e2cc4459485d3d, 0x79d06d5f044e0e1f⟩,
  ⟨0x25f1ddcbd1607b0d, 0x12265f9b8d8868b2⟩,
  ⟨0x57169ceff2335d55, 0xb3961d2b903d11cd⟩,
  ⟨0x51821cfd88fe99a0, 0x3c625c142e1d515d⟩,
  ⟨0x329b9e4fe6f2c912, 0xbb7481dbb408e234⟩,
  ⟨0xafd85dd470c6d22c, 0x990f71d382817816⟩,
  ⟨0x997fe478c0b53f76, 0x26c36b2858dde4cc⟩,
  ⟨0xe9b842f22c8900a4, 0x807b0eb99eb7b556⟩,
  ⟨0xbb97a29817f15f01, 0x12426ef6bda5a89a⟩,
  ⟨0x64dfd5d9402b863, 0x67828ae497510edd⟩,
  ⟨0x3d37f769aee1d2cf, 0xef043cb9fc49b6d0⟩]

theorem svC_step_0 : sIter svC 0 (F128.monomial 0) = svC[0]! := by decide +kernel
theorem svC_step_1 : sIter svC 1 (F128.monomial 1) = svC[1]! := by decide +kernel
theorem svC_step_2 : sIter svC 2 (F128.monomial 2) = svC[2]! := by decide +kernel
theorem svC_step_3 : sIter svC 3 (F128.monomial 3) = svC[3]! := by decide +kernel
theorem svC_step_4 : sIter svC 4 (F128.monomial 4) = svC[4]! := by decide +kernel
theorem svC_step_5 : sIter svC 5 (F128.monomial 5) = svC[5]! := by decide +kernel
theorem svC_step_6 : sIter svC 6 (F128.monomial 6) = svC[6]! := by decide +kernel
theorem svC_step_7 : sIter svC 7 (F128.monomial 7) = svC[7]! := by decide +kernel
theorem svC_step_8 : sIter svC 8 (F128.monomial 8) = svC[8]! := by decide +kernel
theorem svC_step_9 : sIter svC 9 (F128.monomial 9) = svC[9]! := by decide +kernel
theorem svC_step_10 : sIter svC 10 (F128.monomial 10) = svC[10]! := by decide +kernel
theorem svC_step_11 : sIter svC 11 (F128.monomial 11) = svC[11]! := by decide +kernel
theorem svC_step_12 : sIter svC 12 (F128.monomial 12) = svC[12]! := by decide +kernel
theorem svC_step_13 : sIter svC 13 (F128.monomial 13) = svC[13]! := by decide +kernel
theorem svC_step_14 : sIter svC 14 (F128.monomial 14) = svC[14]! := by decide +kernel
theorem svC_step_15 : sIter svC 15 (F128.monomial 15) = svC[15]! := by decide +kernel
theorem svC_step_16 : sIter svC 16 (F128.monomial 16) = svC[16]! := by decide +kernel
theorem svC_step_17 : sIter svC 17 (F128.monomial 17) = svC[17]! := by decide +kernel
theorem svC_step_18 : sIter svC 18 (F128.monomial 18) = svC[18]! := by decide +kernel
theorem svC_step_19 : sIter svC 19 (F128.monomial 19) = svC[19]! := by decide +kernel
theorem svC_step_20 : sIter svC 20 (F128.monomial 20) = svC[20]! := by decide +kernel
theorem svC_step_21 : sIter svC 21 (F128.monomial 21) = svC[21]! := by decide +kernel
theorem svC_step_22 : sIter svC 22 (F128.monomial 22) = svC[22]! := by decide +kernel
theorem svC_step_23 : sIter svC 23 (F128.monomial 23) = svC[23]! := by decide +kernel
theorem svC_step_24 : sIter svC 24 (F128.monomial 24) = svC[24]! := by decide +kernel
theorem svC_step_25 : sIter svC 25 (F128.monomial 25) = svC[25]! := by decide +kernel
theorem svC_step_26 : sIter svC 26 (F128.monomial 26) = svC[26]! := by decide +kernel
theorem svC_step_27 : sIter svC 27 (F128.monomial 27) = svC[27]! := by decide +kernel
theorem svC_step_28 : sIter svC 28 (F128.monomial 28) = svC[28]! := by decide +kernel
theorem svC_step_29 : sIter svC 29 (F128.monomial 29) = svC[29]! := by decide +kernel
theorem svC_step_30 : sIter svC 30 (F128.monomial 30) = svC[30]! := by decide +kernel
theorem svC_step_31 : sIter svC 31 (F128.monomial 31) = svC[31]! := by decide +kernel
theorem svC_step_32 : sIter svC 32 (F128.monomial 32) = svC[32]! := by decide +kernel
theorem svC_step_33 : sIter svC 33 (F128.monomial 33) = svC[33]! := by decide +kernel
theorem svC_step_34 : sIter svC 34 (F128.monomial 34) = svC[34]! := by decide +kernel
theorem svC_step_35 : sIter svC 35 (F128.monomial 35) = svC[35]! := by decide +kernel
theorem svC_step_36 : sIter svC 36 (F128.monomial 36) = svC[36]! := by decide +kernel
theorem svC_step_37 : sIter svC 37 (F128.monomial 37) = svC[37]! := by decide +kernel
theorem svC_step_38 : sIter svC 38 (F128.monomial 38) = svC[38]! := by decide +kernel
theorem svC_step_39 : sIter svC 39 (F128.monomial 39) = svC[39]! := by decide +kernel

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
