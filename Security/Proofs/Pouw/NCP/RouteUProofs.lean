import Models.Pouw.Assumptions.NCP.RouteUGame
import Proofs.Pouw.NCP.GameProofs
import Proofs.Pouw.Sanity
import Proofs.Pouw.NCP.ChainProofs
import Models.Pouw.Guarantees.NCP

/-!
# NCP's checked values on route U: proofs

* `checkedU_eq` (the shift identity), `checkedU_final` (`C^U_T = A·B`), `biasU8Iff` and `rangeU` (u8 × s8),
  `mmWU_eq` and `wrapU` (the int32 reading), `int7Int32U` (signed `R`, `k ≤ 2^15`), `int7Int32UUnsigned`
  (permutation matrix, `k ≤ 2^16`: `runU_cells`, `cell_bound`), `mmWUExact`, `leadCompleteU` and `yuChecked`.
-/

namespace Pouw.NCP.RouteUProofs

open Finset Pouw.Proofs Pouw.NCP.RouteU

theorem accWFrom_cast (c : ℤ) (f : ℕ → ℤ) (n : ℕ) :
    ((accWFrom c f n : ℤ) : Z32) = (c : Z32) + ∑ l ∈ range n, ((f l : ℤ) : Z32) := by
  induction n with
  | zero => simp [accWFrom, wrap32_cast]
  | succ n ih => rw [accWFrom, wrap32_cast, Int.cast_add, ih, Finset.sum_range_succ, add_assoc]

theorem accWFrom_eq (c : ℤ) (f : ℕ → ℤ) (n : ℕ) : accWFrom c f n = wrap32 (c + ∑ l ∈ range n, f l) := by
  refine eq_of_cast_eq ?_ (wrap32_mem _) ?_
  · cases n with
    | zero => exact wrap32_mem _
    | succ n => exact wrap32_mem _
  · rw [accWFrom_cast, wrap32_cast, Int.cast_add, Int.cast_sum]

/-- **The int32 reading.** -/
theorem mmWU_eq : MmWUEq := by
  intro β k A B E1 F1 R τ i j
  rw [mmWU, accWFrom_eq]
  rfl

/-- Route U's running value is the unbiased one plus `β·(Σ_{l < τ} Y − Σ_{l < 3k} Y)`. -/
theorem runU_eq (β : ℤ) (k : ℕ) (A B E1 F1 R : Mat) (τ i j : ℕ) :
    runU β k A B E1 F1 R τ i j = mm τ (stackX k A E1 R) (stackY k B R F1) i j +
      β * (∑ l ∈ range τ, stackY k B R F1 l j - ∑ l ∈ range (3 * k), stackY k B R F1 l j) := by
  simp only [runU, initU, mm, stackXU, add_mul, sum_add_distrib, ← mul_sum]
  ring

/-- **The shift identity.** -/
theorem checkedU_eq : Pouw.Guarantees.NCP.CheckedUEq := by
  intro β k A B E1 F1 R t i j ht
  rw [checkedU, runU_eq, ← sum_range_add_sum_Ico _ ht]
  ring

/-- At depth `3k` route U's running value is the stacked product, so `A·B`. -/
theorem runU_full (β : ℤ) (k : ℕ) (A B E1 F1 R : Mat) (i j : ℕ) :
    runU β k A B E1 F1 R (3 * k) i j = mm k A B i j := by
  rw [runU_eq, sub_self, mul_zero, add_zero]
  exact congrFun (congrFun (NCP.Proofs.stacked k A B E1 F1 R).2 i) j

theorem sixteen_T {k : ℕ} (h : 16 ∣ k) : 16 * (3 * k / 16) = 3 * k := by omega

/-- **The final checked value.** -/
theorem checkedU_final : Pouw.Guarantees.NCP.CheckedUFinal := by
  intro β k A B E1 F1 R i j h
  rw [checkedU, sixteen_T h, runU_full]

/-! ## The operand ranges -/

theorem unsigned_signed {k : ℕ} {R : Mat} (hR : Unsigned k R) : SignedPerm k R := by
  obtain ⟨π, hπ⟩ := hR
  exact ⟨π, fun _ => 1, fun _ => Or.inl rfl, hπ⟩

/-- For a permutation matrix, `E₂[i, l] = E₁[i, π l]`. -/
theorem e2_unsigned {k : ℕ} {E1 R : Mat} (hR : Unsigned k R) (i : ℕ) {l : ℕ} (hl : l < k) :
    ∃ x < k, E2 k E1 R i l = E1 i x := by
  obtain ⟨π, hR⟩ := hR
  have hc : ∀ x : Fin k, R x l = if x = π ⟨l, hl⟩ then 1 else 0 := fun x => hR x ⟨l, hl⟩
  refine ⟨π ⟨l, hl⟩, (π ⟨l, hl⟩).isLt, ?_⟩
  simp only [E2, mm]
  rw [Finset.sum_range (fun x => E1 i x * R x l)]
  simp only [hc, mul_ite, mul_one, mul_zero, Finset.sum_ite_eq', Finset.mem_univ, ite_true]

theorem stackX_at (k : ℕ) (A E1 R : Mat) (i : ℕ) {a : ℕ} (ha : a < k) :
    stackX k A E1 R i a = A i a + E1 i a ∧ stackX k A E1 R i (k + a) = A i a + E2 k E1 R i a ∧
      stackX k A E1 R i (k + k + a) = -(A i a + (E1 i a + E2 k E1 R i a)) := by
  simp only [stackX, hcat3, E3, Pi.add_apply, Pi.neg_apply]
  refine ⟨by simp [ha], ?_, ?_⟩
  · rw [if_neg (by omega), if_pos (by omega), Nat.add_sub_cancel_left]
  · rw [if_neg (by omega), if_neg (by omega), show k + k + a - 2 * k = a by omega]

theorem stackY_at (k : ℕ) (B R F1 : Mat) {a : ℕ} (ha : a < k) (j : ℕ) :
    stackY k B R F1 a j = B a j + F1 a j ∧ stackY k B R F1 (k + a) j = B a j + F2 k R F1 a j ∧
      stackY k B R F1 (k + k + a) j = B a j + (F1 a j + F2 k R F1 a j) := by
  simp only [stackY, vcat3, F3, Pi.add_apply]
  refine ⟨by simp [ha], ?_, ?_⟩
  · rw [if_neg (by omega), if_pos (by omega), Nat.add_sub_cancel_left]
  · rw [if_neg (by omega), if_neg (by omega), show k + k + a - 2 * k = a by omega]

/-- Every `l < 3k` is `a`, `k + a` or `k + k + a` for some `a < k`. -/
theorem three_cases {k l : ℕ} (hl : l < 3 * k) : ∃ a < k, l = a ∨ l = k + a ∨ l = k + k + a := by
  by_cases h1 : l < k
  · exact ⟨l, h1, Or.inl rfl⟩
  by_cases h2 : l < 2 * k
  · exact ⟨l - k, by omega, Or.inr (Or.inl (by omega))⟩
  · exact ⟨l - 2 * k, by omega, Or.inr (Or.inr (by omega))⟩

/-- For a permutation matrix, `X` is in `[−127, 126]` on the int7 domain. -/
theorem stackX_unsigned {k : ℕ} {A E1 R : Mat} (hR : Unsigned k R) (hE : ∀ i, ∀ l < k, InNoise6 (E1 i l))
    (hA : ∀ i, ∀ l < k, InInt7 (A i l)) (i : ℕ) {l : ℕ} (hl : l < 3 * k) :
    -127 ≤ stackX k A E1 R i l ∧ stackX k A E1 R i l ≤ 126 := by
  obtain ⟨a, ha, h⟩ := three_cases hl
  obtain ⟨x1, x2, x3⟩ := stackX_at k A E1 R i ha
  obtain ⟨x, hx, e2⟩ := e2_unsigned (E1 := E1) hR i ha
  have := hA i a ha; have := hE i a ha; have := hE i x hx
  unfold InInt7 InNoise6 at *
  rcases h with rfl | rfl | rfl
  · rw [x1]; constructor <;> linarith
  · rw [x2, e2]; constructor <;> linarith
  · rw [x3, e2]; constructor <;> linarith

/-- For a signed permutation, `X` is in `[−127, 127]` on the int7 domain (`GameProofs.rangeLead`). -/
theorem stackX_signed {k : ℕ} {A E1 R : Mat} (hR : SignedPerm k R) (hE : ∀ i, ∀ l < k, InNoise6 (E1 i l))
    (hA : ∀ i, ∀ l < k, InInt7 (A i l)) (i : ℕ) {l : ℕ} (hl : l < 3 * k) :
    -127 ≤ stackX k A E1 R i l ∧ stackX k A E1 R i l ≤ 127 :=
  (GameProofs.rangeLead k A 0 E1 0 R hR hE (fun _ _ _ => by unfold InNoise6; norm_num) hA
    (fun _ _ _ => by unfold InInt7; norm_num)).1 i l hl

/-- The identity on `{0}` is a permutation matrix. -/
theorem unsigned_one : Unsigned 1 (fun i j => if i = j then 1 else 0) :=
  ⟨Equiv.refl _, fun i j => by simp [Fin.ext_iff]⟩

/-- The swap on `{0, 1}` with both signs `−1` is a signed permutation. -/
theorem signed_swap : SignedPerm 2 (fun i j => if (i = 0 ∧ j = 1) ∨ (i = 1 ∧ j = 0) then -1 else 0) := by
  refine ⟨Equiv.swap 0 1, fun _ => -1, fun _ => Or.inr rfl, fun i j => ?_⟩
  fin_cases i <;> fin_cases j <;> simp [Equiv.swap_apply_def]

/-- **The biases that keep `X + β` in u8.** -/
theorem biasU8Iff : Pouw.Guarantees.NCP.BiasU8Iff := by
  intro β
  have lo : ∀ Rs : ℕ → Mat → Prop, Rs 1 (fun i j => if i = j then 1 else 0) → BiasU8 β Rs →
      127 ≤ β ∧ β ≤ 129 := by
    intro Rs h1 hb
    have hlo := (hb 1 (fun _ _ => 63) (fun _ _ => 32) _ h1 (fun _ _ _ => by unfold InNoise6; norm_num)
      (fun _ _ _ => by unfold InInt7; norm_num) 0 2 (by norm_num)).1
    have hhi := (hb 1 (fun _ _ => -64) (fun _ _ => -31) _ h1 (fun _ _ _ => by unfold InNoise6; norm_num)
      (fun _ _ _ => by unfold InInt7; norm_num) 0 2 (by norm_num)).2
    simp [stackXU, stackX, hcat3, E3, E2, mm] at hlo hhi
    constructor <;> linarith
  refine ⟨⟨lo Unsigned unsigned_one, fun ⟨h1, h2⟩ k A E1 R hR hE hA i l hl => ?_⟩,
    ⟨fun hb => ⟨(lo SignedPerm (unsigned_signed unsigned_one) hb).1, ?_⟩,
      fun ⟨h1, h2⟩ k A E1 R hR hE hA i l hl => ?_⟩⟩
  · have := stackX_unsigned hR hE hA i hl
    simp only [stackXU]
    constructor <;> linarith [this.1, this.2]
  · have h := (hb 2 (fun _ _ => -64) (fun _ l => if l = 0 then -31 else 32) _ signed_swap
      (fun _ l _ => by unfold InNoise6; split_ifs <;> norm_num) (fun _ _ _ => by unfold InInt7; norm_num) 0 4
      (by norm_num)).2
    simp [stackXU, stackX, hcat3, E3, E2, mm, Finset.sum_range_succ] at h
    linarith
  · have := stackX_signed hR hE hA i hl
    simp only [stackXU]
    constructor <;> linarith [this.1, this.2]

/-- **Route U's operand ranges.** -/
theorem rangeU : Pouw.Guarantees.NCP.RangeU := by
  intro β k A B E1 F1 R h1 h2 hR hE hF hA hB
  obtain ⟨-, hY⟩ := GameProofs.rangeLead k A B E1 F1 R (unsigned_signed hR) hE hF hA hB
  refine ⟨fun i l hl => ?_, fun l j hl => ?_⟩
  · have := stackX_unsigned hR hE hA i hl
    simp only [stackXU]
    constructor <;> linarith [this.1, this.2]
  · have := hY l j hl
    unfold InInt8
    constructor <;> linarith [this.1, this.2]

/-! ## The int32 reading -/

/-- **int32 exactness of the last checkpoint.** -/
theorem wrapU : Pouw.Guarantees.NCP.WrapU := by
  intro β k A B E1 F1 R i j h hk hA hB
  have hm : InInt32 (mm k A B i j) := int7_sum_mem k (fun l => A i l) (fun l => B l j) hk hA hB
  have e : wrap32 (checkedU β k A B E1 F1 R (3 * k / 16) i j) = mm k A B i j := by
    rw [checkedU_final β k A B E1 F1 R i j h, wrap32_of_mem hm]
  refine ⟨?_, e⟩
  rw [mmWU_eq, ← e, checkedU]

/-- The int32 accumulator at `3k` is `A·B` for int7 `A`, `B` and `k ≤ 2^19 − 1` (no `16 ∣ k` needed). -/
theorem mmWU_full {k : ℕ} {A B : Mat} (β : ℤ) (E1 F1 R : Mat) {i j : ℕ} (hk : k ≤ 2 ^ 19 - 1)
    (hA : ∀ l < k, InInt7 (A i l)) (hB : ∀ l < k, InInt7 (B l j)) :
    mmWU β k A B E1 F1 R (3 * k) i j = mm k A B i j := by
  have hm : InInt32 (mm k A B i j) := int7_sum_mem k (fun l => A i l) (fun l => B l j) hk hA hB
  rw [mmWU_eq, runU_full, wrap32_of_mem hm]

theorem abs_prod {x y c d : ℤ} (hx : |x| ≤ c) (hy : |y| ≤ d) : -(c * d) ≤ x * y ∧ x * y ≤ c * d := by
  refine abs_le.1 ?_
  rw [abs_mul]
  exact mul_le_mul hx hy (abs_nonneg _) (le_trans (abs_nonneg _) hx)

theorem abs_beta {β : ℤ} (h1 : 127 ≤ β) (h2 : β ≤ 129) : |β| ≤ 129 := by
  rw [abs_of_nonneg (by linarith)]
  exact h2

/-- **The int32 range on route U, signed `R`.** -/
theorem int7Int32U : Int7Int32U := by
  intro β k A B E1 F1 R τ i j hβ1 hβ2 hR hE hF hA hB hk hτ
  obtain ⟨hX, hY⟩ := GameProofs.rangeLead k A B E1 F1 R hR hE hF hA hB
  have e : runU β k A B E1 F1 R τ i j = ∑ l ∈ range τ, stackX k A E1 R i l * stackY k B R F1 l j
      - β * ∑ l ∈ range (3 * k - τ), stackY k B R F1 (τ + l) j := by
    rw [runU_eq, ← sum_range_add_sum_Ico _ hτ, sum_Ico_eq_sum_range]
    simp only [mm]
    ring
  have h1 := sum_bounds τ (fun l => stackX k A E1 R i l * stackY k B R F1 l j) (-(127 * 127)) (127 * 127)
    fun l hl => GameProofs.mul_bound (hX i l (by omega)) (hY l j (by omega))
  have h2 := sum_bounds (3 * k - τ) (fun l => stackY k B R F1 (τ + l) j) (-127) 127
    fun l hl => hY _ j (by omega)
  have hab : (τ : ℤ) + ((3 * k - τ : ℕ) : ℤ) = 3 * k := by
    rw [Nat.cast_sub hτ]
    push_cast
    ring
  have hkz : (k : ℤ) ≤ 32768 := by exact_mod_cast (show k ≤ 32768 by norm_num at hk; omega)
  have ha : (0 : ℤ) ≤ τ := Nat.cast_nonneg _
  have hb : (0 : ℤ) ≤ ((3 * k - τ : ℕ) : ℤ) := Nat.cast_nonneg _
  have h2' : |∑ l ∈ range (3 * k - τ), stackY k B R F1 (τ + l) j| ≤ ((3 * k - τ : ℕ) : ℤ) * 127 :=
    abs_le.2 ⟨by linarith [h2.1], h2.2⟩
  have hβS := abs_prod (abs_beta hβ1 hβ2) h2'
  rw [e]
  unfold InInt32
  constructor <;> nlinarith [h1.1, h1.2, hβS.1, hβS.2]

/-! ## The int32 range for a permutation matrix -/

/-- For a permutation matrix, `F₂[l, j] = −F₁[π⁻¹ l, j]`. -/
theorem f2_unsigned {k : ℕ} {R F1 : Mat} (hR : Unsigned k R) {l : ℕ} (hl : l < k) (j : ℕ) :
    ∃ x < k, F2 k R F1 l j = -F1 x j := by
  obtain ⟨π, hR⟩ := hR
  have hc : ∀ x : Fin k, R l x = if x = π.symm ⟨l, hl⟩ then 1 else 0 := fun x => by
    have hiff : ((⟨l, hl⟩ : Fin k) = π x) ↔ x = π.symm ⟨l, hl⟩ :=
      ⟨fun h => by rw [h, Equiv.symm_apply_apply], fun h => by rw [h, Equiv.apply_symm_apply]⟩
    exact (hR ⟨l, hl⟩ x).trans (if_congr hiff rfl rfl)
  refine ⟨π.symm ⟨l, hl⟩, (π.symm ⟨l, hl⟩).isLt, ?_⟩
  simp only [F2, mm]
  rw [Finset.sum_range (fun x => R l x * F1 x j)]
  simp only [hc, ite_mul, one_mul, zero_mul, Finset.sum_ite_eq', Finset.mem_univ, ite_true]

/-- For a permutation matrix, `Σ_l F₂[l, j] = −Σ_l F₁[l, j]` (the column sums of `R` are `1`). -/
theorem sum_F2_unsigned {k : ℕ} {R F1 : Mat} (hR : Unsigned k R) (j : ℕ) :
    ∑ a ∈ range k, F2 k R F1 a j = -∑ a ∈ range k, F1 a j := by
  obtain ⟨π, hR⟩ := hR
  have hcol : ∀ x < k, ∑ a ∈ range k, R a x = 1 := by
    intro x hx
    have hc : ∀ a : Fin k, R a x = if a = π ⟨x, hx⟩ then 1 else 0 := fun a => hR a ⟨x, hx⟩
    rw [Finset.sum_range (fun a => R a x)]
    simp only [hc, Finset.sum_ite_eq', Finset.mem_univ, ite_true]
  simp only [F2, mm, sum_neg_distrib]
  rw [sum_comm]
  congr 1
  refine sum_congr rfl fun x hx => ?_
  rw [← sum_mul, hcol x (mem_range.1 hx), one_mul]

theorem sum_three (k : ℕ) (g : ℕ → ℤ) :
    ∑ l ∈ range (3 * k), g l = ∑ a ∈ range k, (g a + g (k + a) + g (k + k + a)) := by
  rw [show 3 * k = k + k + k by ring, sum_range_add, sum_range_add, sum_add_distrib, sum_add_distrib]

/-- Route U's running value as one sum over the `3k` stacked terms, `c₀` shared out. -/
theorem runU_terms (β : ℤ) (k : ℕ) (A B E1 F1 R : Mat) {τ : ℕ} (hτ : τ ≤ 3 * k) (i j : ℕ) :
    runU β k A B E1 F1 R τ i j = ∑ l ∈ range (3 * k),
      ((if l < τ then stackXU β k A E1 R i l else 0) - β) * stackY k B R F1 l j := by
  have hf : (range (3 * k)).filter (· < τ) = range τ := by
    ext l
    simp only [mem_filter, mem_range]
    omega
  simp only [sub_mul, sum_sub_distrib, ite_mul, zero_mul]
  rw [← sum_filter, hf, runU, initU, mm, ← mul_sum]
  ring

/-- One original index's three stacked terms, moved by `c·β·(F₁ + F₂)`, are at most `29,856` in magnitude. -/
theorem cell_bound {β x e1 e2 b f1 f2 : ℤ} (hβ1 : 127 ≤ β) (hβ2 : β ≤ 129) (hx : |x| ≤ 64) (he1 : |e1| ≤ 32)
    (he2 : |e2| ≤ 32) (hb : |b| ≤ 64) (hf1 : |f1| ≤ 32) (hf2 : |f2| ≤ 32) {τ a k : ℕ} (ha : a < k) :
    |((if a < τ then x + e1 + β else 0) - β) * (b + f1) + ((if k + a < τ then x + e2 + β else 0) - β) * (b + f2) +
        ((if k + k + a < τ then -(x + (e1 + e2)) + β else 0) - β) * (b + (f1 + f2)) +
        (if τ ≤ k then 2 else 1) * β * (f1 + f2)| ≤ 29856 := by
  have hβ := abs_beta hβ1 hβ2
  have s1 : |x + e1| ≤ 96 := (abs_add_le _ _).trans (by linarith)
  have s2 : |x + e2| ≤ 96 := (abs_add_le _ _).trans (by linarith)
  have t1 : |b + f1| ≤ 96 := (abs_add_le _ _).trans (by linarith)
  have t2 : |b + f2| ≤ 96 := (abs_add_le _ _).trans (by linarith)
  have p1 := abs_prod s1 t1
  have p2 := abs_prod s2 t2
  have pab := abs_prod hx hb
  have p12 := abs_prod he1 hf2
  have p21 := abs_prod he2 hf1
  have pb := abs_prod hβ hb
  have pf1 := abs_prod hβ hf1
  have pf2 := abs_prod hβ hf2
  rw [abs_le]
  split_ifs <;> first | omega | (constructor <;> linarith [p1.1, p1.2, p2.1, p2.2, pab.1, pab.2, p12.1, p12.2,
    p21.1, p21.2, pb.1, pb.2, pf1.1, pf1.2, pf2.1, pf2.2])

/-- **The int32 range on route U, permutation-matrix `R`.** -/
theorem int7Int32UUnsigned : Int7Int32UUnsigned := by
  intro β k A B E1 F1 R τ i j hβ1 hβ2 hR hE hF hA hB hk hτ
  have hS := unsigned_signed hR
  have hE' : ∀ l < k, |E1 i l| ≤ 32 := fun l hl => abs_le.2 ⟨by linarith [(hE i l hl).1], (hE i l hl).2⟩
  have hF' : ∀ l < k, |F1 l j| ≤ 32 := fun l hl => abs_le.2 ⟨by linarith [(hF l hl j).1], (hF l hl j).2⟩
  have hzero : ∑ a ∈ range k, (if τ ≤ k then (2 : ℤ) else 1) * β * (F1 a j + F2 k R F1 a j) = 0 := by
    rw [← mul_sum, sum_add_distrib, sum_F2_unsigned hR j, add_neg_cancel, mul_zero]
  set g : ℕ → ℤ := fun l => ((if l < τ then stackXU β k A E1 R i l else 0) - β) * stackY k B R F1 l j with hg
  have e : runU β k A B E1 F1 R τ i j = ∑ a ∈ range k, (g a + g (k + a) + g (k + k + a) +
      (if τ ≤ k then (2 : ℤ) else 1) * β * (F1 a j + F2 k R F1 a j)) := by
    rw [sum_add_distrib, hzero, add_zero, runU_terms β k A B E1 F1 R hτ i j, sum_three]
  have hb := sum_bounds k (fun a => g a + g (k + a) + g (k + k + a) +
      (if τ ≤ k then (2 : ℤ) else 1) * β * (F1 a j + F2 k R F1 a j)) (-29856) 29856 fun a ha => by
    obtain ⟨x1, x2, x3⟩ := stackX_at k A E1 R i ha
    obtain ⟨y1, y2, y3⟩ := stackY_at k B R F1 ha j
    simp only [hg, stackXU, x1, x2, x3, y1, y2, y3]
    exact abs_le.1 (cell_bound hβ1 hβ2 (ChainProofs.abs_int7 (hA i a ha)) (hE' a ha)
      (GameProofs.e2_abs hS hE' ha) (ChainProofs.abs_int7 (hB a ha j)) (hF' a ha) (GameProofs.f2_abs hS hF' ha)
      ha)
  rw [e]
  have hkz : (k : ℤ) ≤ 65536 := by exact_mod_cast (show k ≤ 65536 by norm_num at hk; omega)
  unfold InInt32
  constructor <;> nlinarith [hb.1, hb.2]

/-- **The wrap is the identity at every depth.** -/
theorem mmWUExact : Pouw.Guarantees.NCP.MmWUExact := by
  intro β k A B E1 F1 R τ i j hβ1 hβ2 hR hE hF hA hB hτ
  rw [mmWU_eq]
  refine wrap32_of_mem ?_
  rcases hR with ⟨hR, hk⟩ | ⟨hR, hk⟩
  · exact int7Int32U β k A B E1 F1 R τ i j hβ1 hβ2 hR hE hF hA hB hk hτ
  · exact int7Int32UUnsigned β k A B E1 F1 R τ i j hβ1 hβ2 hR hE hF hA hB hk hτ

/-! ## The protocol on route U -/

/-- **Route U's completeness.** -/
theorem leadCompleteU : LeadCompleteU := by
  intro Q R S P β U hU u hu H s A hA
  obtain ⟨-, ⟨hsh, -⟩, hwt⟩ := hU
  have hk : (U.layout.shape u).k ≤ 2 ^ 19 - 1 := le_trans (hsh u hu).2.2.2.1 (by norm_num)
  refine ⟨fun x _ => rfl, fun i hi j hj => ?_⟩
  exact mmWU_full _ _ _ _ hk (fun l hl => hA _ hi l hl) (fun l hl => hwt u hu l hl _ hj)

/-- **Route U's `Y` at the lead's depths.** -/
theorem yuChecked : Pouw.Guarantees.NCP.YUChecked := by
  intro Q R S P β H s U u A t i j hd h16 ht
  have he : D1.blockEnd P.d (3 * (U.layout.shape u).k) t = 16 * (t + 1) := by
    rw [D1.blockEnd, hd]
    omega
  simp only [Params.YU, he, checkedU]
  exact mmWU_eq _ _ _ _ _ _ _ _ _ _

end Pouw.NCP.RouteUProofs

/-! ## The game instance on route U -/

namespace Pouw.NCP.RouteUProofs

open Finset Pouw.Proofs QTree Pouw.NCP.RouteU

variable {Q R S : Type}

/-- **`TT_NCP_U` is `TT_NCP` of route U's protocol.** -/
theorem ttncpUIff : Pouw.Guarantees.NCP.TTNCPUIff := fun _ _ _ _ _ _ _ _ _ _ _ _ _ => Iff.rfl

theorem wrefU_le (f : ℕ) {n₀ : ℕ} (hn₀ : 0 < n₀) {sh : Shape} (hn : n₀ ≤ sh.n) :
    ((3 * sh.Wmm + f * sh.m * sh.k : ℕ) : ℝ) ≤ (1 + f / (3 * (n₀ : ℝ))) * (3 * (sh.Wmm : ℝ)) := by
  have h0 : (0 : ℝ) < n₀ := by exact_mod_cast hn₀
  have h1 : (n₀ : ℝ) ≤ sh.n := by exact_mod_cast hn
  have h2 : (f : ℝ) * ((sh.m : ℝ) * sh.k) * n₀ ≤ f * ((sh.m : ℝ) * sh.k) * sh.n :=
    mul_le_mul_of_nonneg_left h1 (by positivity)
  have e : (1 + f / (3 * (n₀ : ℝ))) * (3 * (sh.Wmm : ℝ)) = 3 * sh.Wmm + f * ((sh.m : ℝ) * sh.k) * sh.n / n₀ := by
    simp only [Shape.Wmm]
    push_cast
    field_simp
  rw [e, Shape.Wmm]
  push_cast
  have : (f : ℝ) * ((sh.m : ℝ) * sh.k) ≤ f * ((sh.m : ℝ) * sh.k) * sh.n / n₀ := by
    rw [le_div_iff₀ h0]
    exact h2
  linarith

/-- **`G_γ` from TT_NCP_U.** -/
theorem gammaFromTTNCP_U : GammaFromTTNCP_U := by
  intro Q R S _ _ _ _ CM P β f D γ₀ n₀ ε hγ₀ hn₀ hW hn hTT
  have hΩ : (0 : ℝ) < 1 + f / (3 * n₀) := by positivity
  refine GameProofs.gammaFromTTNCP Q R S CM (P.toProtocolU β) D γ₀ (1 + f / (3 * n₀)) ε hγ₀ hΩ
    (fun U hU u hu => ?_) hTT
  show ((P.Wref U.layout u : ℕ) : ℝ) ≤ _
  rw [hW, wrefU]
  exact wrefU_le f hn₀ (hn U hU u hu)

/-- **`G_γ` from TT_NCP_U at ncp-v1.** -/
theorem gammaFromTTNCP_U_v1 : Pouw.Guarantees.NCP.GammaFromTTNCP_U_v1 := by
  intro Q R S _ _ _ _ CM P D γ₀ n₀ ε hγ₀ hn₀ hW hn hTT
  have e : (1 : ℝ) + ((24 : ℕ) : ℝ) / (3 * n₀) = 1 + 8 / n₀ := by
    have : (0 : ℝ) < n₀ := by exact_mod_cast hn₀
    field_simp
    push_cast
    ring
  rw [← e]
  exact gammaFromTTNCP_U Q R S CM P 128 24 D γ₀ n₀ ε hγ₀ hn₀ hW hn hTT

/-- **`G_γ` from TT_NCP_U at C2.** -/
theorem gammaFromTTNCP_U_C2 : Pouw.Guarantees.NCP.GammaFromTTNCP_U_C2 := by
  intro Q R S _ _ _ _ CM P D γ₀ n₀ ε hγ₀ hn₀ hW hn hTT
  have e : (1 : ℝ) + ((20 : ℕ) : ℝ) / (3 * n₀) = 1 + 20 / (3 * n₀) := by norm_num
  rw [← e]
  exact gammaFromTTNCP_U Q R S CM P 129 20 D γ₀ n₀ ε hγ₀ hn₀ hW hn hTT

section Honest

variable (P : Params Q R S) (β : ℤ)

/-- Unit `u`'s honest commitment on route U from the answers `a` to `qAct` and `b` to `qWt`. -/
def commitAnsU (U : Workload) (u : ℕ) (A : Mat) (a b : List R) : Commit :=
  let sh := U.layout.shape u
  let B := U.weight (U.layout.wid u)
  ⟨A, fun t => mmWU β sh.k A B (P.expAct sh a) (P.expWt sh b).1 (P.expWt sh b).2 (D1.blockEnd P.d (3 * sh.k) t),
    mmWU β sh.k A B (P.expAct sh a) (P.expWt sh b).1 (P.expWt sh b).2 (3 * sh.k)⟩

theorem commitAnsU_eq (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) :
    commitAnsU P β U u A ((P.qAct s u A (U.layout.wid u)).map H) ((P.qWt s (U.layout.wid u)).map H)
      = ⟨A, (P.toProtocolU β).refChecked H s U u A, (P.toProtocolU β).refOut H s U u A⟩ := rfl

/-- Unit `u`'s honest program on route U: query its noise, then commit. -/
def unitProgU (U : Workload) (act : ℕ → Mat) (s : S) (u : ℕ) : QTree Q R Commit :=
  (queryAll (P.qAct s u (act u) (U.layout.wid u))).bind fun a =>
    (queryAll (P.qWt s (U.layout.wid u))).bind fun b => done (commitAnsU P β U u (act u) a b)

/-- Route U's honest reference as an oracle program. -/
def honestProgU (U : Workload) (act : ℕ → Mat) (s : S) : QTree Q R Transcript :=
  upTo (unitProgU P β U act s) U.layout.N

theorem eval_honestProgU (H : Q → R) (U : Workload) (act : ℕ → Mat) (s : S) :
    (honestProgU P β U act s).eval H = honestTranscript (P.toProtocolU β) H s U act := by
  unfold honestProgU honestTranscript
  rw [eval_upTo]
  funext u
  simp only [unitProgU, eval_bind, eval_queryAll, eval, commitAnsU_eq]

theorem calls_honestProgU (H : Q → R) (U : Workload) (act : ℕ → Mat) (s : S) {c : ℕ} (hc : P.QueryBound c) :
    (honestProgU P β U act s).calls H ≤ U.layout.N * c := by
  refine calls_upTo_le H _ _ c fun u _ => ?_
  simp only [unitProgU, calls_bind, calls_queryAll, eval_queryAll, calls]
  have := hc s u (act u) (U.layout.wid u)
  omega

end Honest

/-- **TT_NCP_U's satisfiability witness.** -/
theorem ttncpUWitness : Pouw.Guarantees.NCP.TTNCPUWitness := by
  intro Q R _ _ _ P β c γ₀ h0 h1 hc hW
  refine ⟨GameProofs.ttncp_outputPriced3 _ _ h0 h1, fun U act _ s => honestProgU P β U act s, fun U hU act _ H s =>
    ⟨eval_honestProgU P β H U act s, ?_, calls_honestProgU P β H U act s hc⟩⟩
  refine (GameProofs.outputPriced3_cost_le _ H U act [] s).trans (Finset.sum_le_sum fun u hu => ?_)
  exact_mod_cast hW U hU u (Finset.mem_range.1 hu)

/-- `shiftBy c k` is a permutation matrix. -/
theorem shiftBy_unsigned {c k : ℕ} (hc : c < k) : Unsigned k (GameProofs.shiftBy c k) := by
  obtain ⟨n, rfl⟩ : ∃ n, k = n + 1 := ⟨k - 1, by omega⟩
  refine ⟨Equiv.addRight (⟨c, hc⟩ : Fin (n + 1)), fun i j => ?_⟩
  have h : ((Equiv.addRight (⟨c, hc⟩ : Fin (n + 1)) j : Fin (n + 1)) : ℕ)
      = if (j : ℕ) + c < n + 1 then (j : ℕ) + c else (j : ℕ) + c - (n + 1) := by
    simp only [Equiv.coe_addRight, Fin.val_add]
    split_ifs with hj
    · exact Nat.mod_eq_of_lt hj
    · rw [Nat.mod_eq_sub_mod (by omega), Nat.mod_eq_of_lt (by omega)]
  by_cases hij : i = Equiv.addRight (⟨c, hc⟩ : Fin (n + 1)) j
  · simp only [GameProofs.shiftBy, j.isLt, true_and, hij, h, ite_true]
  · have : ¬ ((i : ℕ) = if (j : ℕ) + c < n + 1 then (j : ℕ) + c else (j : ℕ) + c - (n + 1)) :=
      fun e => hij (Fin.ext (e.trans h.symm))
    simp only [GameProofs.shiftBy, j.isLt, true_and, this, hij, ite_false]

/-- The route-U witness instance: `leadParams` with `W_ref = wrefU f`. -/
def leadParamsU (f : ℕ) : Params Unit Unit Salt :=
  ⟨16, fun _ _ _ _ => [], fun _ _ => [], fun _ _ => 0, fun sh _ => (0, GameProofs.shiftBy 24 sh.k), wrefU f⟩

/-- **The hypotheses of `GammaFromTTNCP_U` hold jointly at the lead.** -/
theorem leadUWitness : LeadUWitness := by
  intro β f
  have hN : (leadParamsU f).NoiseRange :=
    ⟨fun _ _ _ _ _ _ => by simp [leadParamsU, InNoise6], fun _ _ _ _ _ _ => by simp [leadParamsU, InNoise6]⟩
  have hadm : (leadParamsU f).RAdmissible := fun sh _ hsh =>
    GameProofs.shiftBy24_admissible (le_of_eq_of_le (by norm_num) hsh.2.2.1)
      (Nat.dvd_trans (by norm_num) hsh.2.2.2.2.1)
  have hS : (leadParamsU f).RSigned := fun sh b hsh => (hadm sh b hsh).1
  have hU : ∀ (sh : Shape) (b : List Unit), Shape.InDefaultDomain sh →
      Unsigned sh.k ((leadParamsU f).expWt sh b).2 := fun sh _ hsh =>
    shiftBy_unsigned (lt_of_lt_of_le (by norm_num) hsh.2.2.1)
  have hq : (leadParamsU f).QueryBound 0 := fun _ _ _ _ => by simp [leadParamsU]
  have hW : ∀ U : Workload, U.InDomain leadDomain → ∀ u < U.layout.N,
      3 * (U.layout.shape u).Wmm ≤ (leadParamsU f).Wref U.layout u := fun _ _ _ _ => by
    simp only [leadParamsU, wrefU]
    omega
  have hadmit := (ttncpUWitness Unit Unit (leadParamsU f) β 0 0 le_rfl one_pos hq hW).2
  simp only [mul_zero] at hadmit
  exact ⟨leadParamsU f, GameProofs.leadLayout, ⟨rfl, hN, hadm⟩, hS, hU, hq, fun _ _ => rfl,
    leadCompleteU Unit Unit Salt (leadParamsU f) β, hadmit,
    fun γ₀ h0 h1 => (ttncpUWitness Unit Unit (leadParamsU f) β 0 γ₀ h0 h1 hq hW).1,
    fun U hU u hu => (hU.2.1.1 u hu).2.2.2.2.2.1, GameProofs.leadLayout_inDomain, one_pos, fun _ _ => ⟨64, rfl⟩⟩

end Pouw.NCP.RouteUProofs
