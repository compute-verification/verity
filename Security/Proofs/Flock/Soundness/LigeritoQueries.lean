import Proofs.Flock.Soundness.LigeritoCoins
import Mathlib.Analysis.MeanInequalities

/-!
# Ligerito's stratified queries (PROTOCOL.md §13.5)

If a set `D` covers a fraction `δ` of a level's `2^d` positions, the `Q` stratified query
positions all miss it with probability at most `(1 − δ)^Q`, the same as `Q` independent uniform
positions (`prCoin_positions_miss_le`):
* the strata sum to `Q` (`strata_sum`), and each query is uniform in its stratum's block, since the
  low bits of `lo` are balanced (`Arith.Correct.lo_balanced`);
* the queries' coins are independent, so the miss probabilities multiply (`prCoin_forall_fin`);
* within one summand the blocks partition the positions, so AM–GM bounds the product of the
  blocks' miss probabilities by `(1 − δ)^{2^c}`.
-/

namespace FlockSoundness

open Game Model Accounting
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-! ## The strata -/

theorem strata_go_sum (d : ℕ) : ∀ fuel rem, rem ≤ fuel →
    ((strata.go d fuel rem).map (2 ^ ·)).sum = rem
  | 0, rem, h => by simp [strata.go]; omega
  | fuel + 1, rem, h => by
      rw [strata.go]
      split_ifs with h0
      · simp [h0]
      · simp only [List.map_cons, List.sum_cons]
        have hc : 2 ^ min (Nat.log 2 rem) d ≤ rem :=
          (Nat.pow_le_pow_right (by norm_num) (min_le_left _ _)).trans (Nat.pow_log_le_self 2 h0)
        have hc1 : 1 ≤ 2 ^ min (Nat.log 2 rem) d := Nat.one_le_two_pow
        rw [strata_go_sum d fuel (rem - 2 ^ min (Nat.log 2 rem) d) (by omega)]
        omega

/-- The strata's sizes sum to `Q`. -/
theorem strata_sum (Q d : ℕ) : ((strata Q d).map (2 ^ ·)).sum = Q :=
  strata_go_sum d Q Q le_rfl

theorem strata_go_le (d : ℕ) : ∀ fuel rem, ∀ c ∈ strata.go d fuel rem, c ≤ d
  | 0, rem => by simp [strata.go]
  | fuel + 1, rem => by
      rw [strata.go]
      split_ifs with h0
      · simp
      · intro c hc
        simp only [List.mem_cons] at hc
        rcases hc with rfl | hc
        · exact min_le_right _ _
        · exact strata_go_le d fuel _ c hc

/-- Every stratum's depth is at most `d`. -/
theorem strata_le (Q d : ℕ) : ∀ c ∈ strata Q d, c ≤ d := strata_go_le d Q Q

/-! ## Slots and positions -/

/-- The query slots `(depth, stratum)`, in query order (§13.5). -/
def querySlots (Q d : ℕ) : List (ℕ × ℕ) :=
  (strata Q d).flatMap fun c => (List.range (2 ^ c)).map fun s => (c, s)

theorem querySlots_length (Q d : ℕ) : (querySlots Q d).length = Q := by
  rw [querySlots, List.length_flatMap]
  simpa using strata_sum Q d

/-- The position of the query in slot `(c, s)`, from its coin. -/
def slotPos (A : Arith F K) (d : ℕ) (cs : ℕ × ℕ) (x : F) : ℕ :=
  cs.2 * 2 ^ (d - cs.1) + A.lo x % 2 ^ (d - cs.1)

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem positions_eq (A : Arith F K) (Q d : ℕ) (w : Fin Q → F) :
    positions A Q d w = List.ofFn (fun j : Fin Q =>
      slotPos A d ((querySlots Q d)[j.val]'(by rw [querySlots_length]; exact j.isLt)) (w j)) := by
  unfold positions
  apply List.ext_getElem
  · simpa [querySlots, List.length_flatMap] using strata_sum Q d
  · intro i h1 h2
    simp only [List.getElem_map, List.getElem_zipIdx, List.getElem_ofFn, slotPos, zero_add]
    have hi : i < Q := by simpa using h2
    rw [dif_pos hi]
    rfl

/-! ## Independent coins -/

omit [DecidableEq F] in
/-- Independent coordinates multiply. -/
theorem prCoin_forall_fin : ∀ (Q : ℕ) (E : Fin Q → F → Prop),
    prCoin (fun w : Fin Q → F => ∀ j, E j (w j)) = ∏ j, prCoin (E j)
  | 0, E => by
      classical
      simp only [Finset.univ_eq_empty, Finset.prod_empty, prCoin, avg]
      simp [Fintype.card_fun]
  | Q + 1, E => by
      classical
      rw [prCoin_cons, Fin.prod_univ_succ]
      have hsplit : ∀ a : F, prCoin (fun v : Fin Q → F =>
          ∀ j : Fin (Q + 1), E j ((Fin.cons a v : Fin (Q + 1) → F) j)) =
          (if E 0 a then 1 else 0) * prCoin (fun v : Fin Q → F => ∀ j : Fin Q, E j.succ (v j)) := by
        intro a
        simp only [Fin.forall_fin_succ, Fin.cons_zero, Fin.cons_succ]
        by_cases ha : E 0 a
        · simp [ha]
        · simp [ha, prCoin, avg]
      simp_rw [hsplit, prCoin_forall_fin Q]
      rw [avg_mul_const]
      rfl

/-- **One query.** A query uniform in a block of `2^b` positions (`b ≤ 64`) misses `D` with the
block's fraction of positions outside `D`. -/
theorem prCoin_slot_miss (A : Arith F K) (hA : A.Correct) (b s : ℕ) (hb : b ≤ 64)
    (D : Finset ℕ) :
    prCoin (fun x : F => s * 2 ^ b + A.lo x % 2 ^ b ∉ D) =
      (((Finset.range (2 ^ b)).filter fun t => s * 2 ^ b + t ∉ D).card : ℝ≥0∞) / 2 ^ b := by
  classical
  rw [prCoin_eq_card]
  have hfib : (Finset.univ.filter fun x : F => s * 2 ^ b + A.lo x % 2 ^ b ∉ D).card =
      ∑ t ∈ (Finset.range (2 ^ b)).filter (fun t => s * 2 ^ b + t ∉ D),
        (Finset.univ.filter fun x : F => A.lo x % 2 ^ b = t).card := by
    rw [Finset.card_eq_sum_card_fiberwise (f := fun x => A.lo x % 2 ^ b)
      (t := (Finset.range (2 ^ b)).filter (fun t => s * 2 ^ b + t ∉ D))]
    · refine Finset.sum_congr rfl fun t ht => ?_
      congr 1
      ext x
      simp only [Finset.mem_filter, Finset.mem_univ, true_and]
      constructor
      · rintro ⟨_, h⟩; exact h
      · intro h
        refine ⟨?_, h⟩
        rw [h]
        exact (Finset.mem_filter.1 ht).2
    · intro x hx
      simp only [Finset.coe_filter, Finset.mem_filter, Finset.mem_univ, true_and,
        Set.mem_setOf_eq, Finset.mem_range] at hx ⊢
      exact ⟨Nat.mod_lt _ (Nat.two_pow_pos _), hx⟩
  rw [hfib, Finset.sum_congr rfl fun t ht =>
    hA.lo_balanced b hb t (Finset.mem_range.1 (Finset.mem_filter.1 ht).1),
    Finset.sum_const, smul_eq_mul, hA.card_F]
  have h2 : (2 : ℕ) ^ 128 = 2 ^ b * 2 ^ (128 - b) := by rw [← pow_add]; congr 1; omega
  rw [h2, Nat.mul_div_cancel_left _ (Nat.two_pow_pos _)]
  push_cast
  rw [ENNReal.mul_div_mul_right _ _ (by positivity) (by simp)]

/-! ## AM–GM over the strata -/

/-- AM–GM: a product of `N > 0` nonnegative reals is at most their mean to the `N`. -/
theorem prod_le_mean_pow (N : ℕ) (hN : 0 < N) (z : ℕ → ℝ) (hz : ∀ s ∈ Finset.range N, 0 ≤ z s) :
    ∏ s ∈ Finset.range N, z s ≤ ((∑ s ∈ Finset.range N, z s) / N) ^ N := by
  have h := Real.geom_mean_le_arith_mean (Finset.range N) (fun _ => (1 : ℝ)) z
    (fun _ _ => zero_le_one) (by simp [hN]) hz
  simp only [Real.rpow_one, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one,
    one_mul] at h
  have hP : 0 ≤ ∏ s ∈ Finset.range N, z s := Finset.prod_nonneg hz
  calc ∏ s ∈ Finset.range N, z s
      = ((∏ s ∈ Finset.range N, z s) ^ ((N : ℝ)⁻¹)) ^ N := by
        rw [Real.rpow_inv_natCast_pow hP hN.ne']
    _ ≤ ((∑ s ∈ Finset.range N, z s) / N) ^ N :=
        pow_le_pow_left₀ (Real.rpow_nonneg hP _) h N

/-- Blocks of `B` consecutive positions: the points outside `D` below `M·B`, block by block. -/
theorem sum_block_miss (B : ℕ) (D : Finset ℕ) : ∀ M : ℕ,
    ∑ s ∈ Finset.range M, ((Finset.range B).filter fun t => s * B + t ∉ D).card =
      ((Finset.range (M * B)).filter fun q => q ∉ D).card
  | 0 => by simp
  | M + 1 => by
      rw [Finset.sum_range_succ, sum_block_miss B D M, Finset.card_filter, Finset.card_filter,
        Finset.card_filter, show (M + 1) * B = M * B + B by ring, Finset.sum_range_add]

/-- The fraction of block `(c, s)` of a level with `2^d` positions that lies outside `D`. -/
noncomputable def missFrac (d : ℕ) (D : Finset ℕ) (cs : ℕ × ℕ) : ℝ :=
  (((Finset.range (2 ^ (d - cs.1))).filter fun t => cs.2 * 2 ^ (d - cs.1) + t ∉ D).card : ℝ) /
    2 ^ (d - cs.1)

theorem missFrac_nonneg (d : ℕ) (D : Finset ℕ) (cs : ℕ × ℕ) : 0 ≤ missFrac d D cs := by
  unfold missFrac; positivity

theorem prCoin_slotPos_miss (A : Arith F K) (hA : A.Correct) (d : ℕ) (hd : d ≤ 64)
    (D : Finset ℕ) (cs : ℕ × ℕ) :
    prCoin (fun x : F => slotPos A d cs x ∉ D) = ENNReal.ofReal (missFrac d D cs) := by
  unfold slotPos missFrac
  rw [prCoin_slot_miss A hA (d - cs.1) cs.2 (by omega) D, ENNReal.ofReal_div_of_pos (by positivity),
    ENNReal.ofReal_natCast, ENNReal.ofReal_pow (by norm_num), ENNReal.ofReal_ofNat]

/-- Within one summand of depth `c ≤ d`, the blocks' miss fractions average to `1 − |D|/2^d`. -/
theorem sum_missFrac (d c : ℕ) (hc : c ≤ d) (D : Finset ℕ) (hD : D ⊆ Finset.range (2 ^ d)) :
    ∑ s ∈ Finset.range (2 ^ c), missFrac d D (c, s) = 2 ^ c * (1 - (D.card : ℝ) / 2 ^ d) := by
  unfold missFrac
  simp only
  rw [← Finset.sum_div, ← Nat.cast_sum, sum_block_miss]
  have hpow : 2 ^ c * 2 ^ (d - c) = 2 ^ d := by rw [← pow_add]; congr 1; omega
  rw [hpow]
  have hcard : ((Finset.range (2 ^ d)).filter fun q => q ∉ D).card = 2 ^ d - D.card := by
    rw [Finset.filter_not, Finset.filter_mem_eq_inter, Finset.inter_eq_right.2 hD,
      Finset.card_sdiff_of_subset hD, Finset.card_range]
  rw [hcard]
  have hle : D.card ≤ 2 ^ d := by simpa using Finset.card_le_card hD
  push_cast [hle]
  have h2 : (2 : ℝ) ^ d = 2 ^ c * 2 ^ (d - c) := by rw [← pow_add]; congr 1; omega
  rw [h2]
  have : (2 : ℝ) ^ c ≠ 0 := pow_ne_zero _ two_ne_zero
  have : (2 : ℝ) ^ (d - c) ≠ 0 := pow_ne_zero _ two_ne_zero
  field_simp

theorem prod_range_eq_list {M : Type} [CommMonoid M] (n : ℕ) (f : ℕ → M) :
    ∏ i ∈ Finset.range n, f i = ((List.range n).map f).prod := by
  rw [Finset.prod_eq_multiset_prod, Finset.range_val]
  rfl

theorem list_prod_flatMap {α β : Type} [CommMonoid β] (L : List α) (g : α → List β) :
    (L.flatMap g).prod = (L.map fun a => (g a).prod).prod := by
  induction L with
  | nil => simp
  | cons a L ih => simp [List.flatMap_cons, List.prod_append, ih]

theorem list_prod_map_le {α : Type} (L : List α) (f g : α → ℝ) (hf : ∀ a ∈ L, 0 ≤ f a)
    (hfg : ∀ a ∈ L, f a ≤ g a) : (L.map f).prod ≤ (L.map g).prod := by
  induction L with
  | nil => simp
  | cons a L ih =>
      simp only [List.map_cons, List.prod_cons]
      have h1 := hf a (by simp)
      have h2 := hfg a (by simp)
      have ihL := ih (fun b hb => hf b (by simp [hb])) (fun b hb => hfg b (by simp [hb]))
      have hL0 : 0 ≤ (L.map f).prod :=
        List.prod_nonneg (by simpa using fun b hb => hf b (by simp [hb]))
      exact mul_le_mul h2 ihL hL0 (h1.trans h2)

theorem list_prod_map_pow (L : List ℕ) (x : ℝ) :
    (L.map fun c => x ^ (2 ^ c)).prod = x ^ ((L.map (2 ^ ·)).sum) := by
  induction L with
  | nil => simp
  | cons a L ih => simp [List.prod_cons, List.sum_cons, pow_add, ih]

/-- **The stratified queries.** If `D` covers a fraction `|D|/2^d` of a level's positions, the `Q`
stratified query positions (§13.5) all miss it with probability at most `(1 − |D|/2^d)^Q`. -/
theorem prCoin_positions_miss_le (A : Arith F K) (hA : A.Correct) (Q d : ℕ) (hd : d ≤ 64)
    (D : Finset ℕ) (hD : D ⊆ Finset.range (2 ^ d)) :
    prCoin (fun w : Fin Q → F => ∀ p ∈ positions A Q d w, p ∉ D) ≤
      ENNReal.ofReal ((1 - (D.card : ℝ) / 2 ^ d) ^ Q) := by
  classical
  set sl := querySlots Q d with hsl
  have hlen : sl.length = Q := querySlots_length Q d
  have hevent : ∀ w : Fin Q → F, (∀ p ∈ positions A Q d w, p ∉ D) ↔
      ∀ j : Fin Q, slotPos A d (sl[j.val]'(by rw [hlen]; exact j.isLt)) (w j) ∉ D := by
    intro w
    rw [positions_eq]
    simp only [List.mem_ofFn, forall_exists_index, forall_apply_eq_imp_iff]
    exact Iff.rfl
  simp_rw [hevent]
  rw [prCoin_forall_fin Q (fun j x => slotPos A d (sl[j.val]'(by rw [hlen]; exact j.isLt)) x ∉ D)]
  simp_rw [prCoin_slotPos_miss A hA d hd D]
  rw [← ENNReal.ofReal_prod_of_nonneg (fun j _ => missFrac_nonneg d D _)]
  refine ENNReal.ofReal_le_ofReal ?_
  have hprod : ∏ j : Fin Q, missFrac d D (sl[j.val]'(by rw [hlen]; exact j.isLt)) =
      ((strata Q d).map fun c => ∏ s ∈ Finset.range (2 ^ c), missFrac d D (c, s)).prod := by
    rw [← List.prod_ofFn]
    have hofFn : List.ofFn (fun j : Fin Q =>
        missFrac d D (sl[j.val]'(by rw [hlen]; exact j.isLt))) = sl.map (missFrac d D) := by
      apply List.ext_getElem
      · simp [hlen]
      · intro i h1 h2
        simp
    rw [hofFn, hsl, querySlots, List.map_flatMap, list_prod_flatMap]
    congr 1
    refine List.map_congr_left fun c _ => ?_
    rw [List.map_map, prod_range_eq_list]
    rfl
  rw [hprod]
  have hδ : ∀ c ∈ strata Q d, ∏ s ∈ Finset.range (2 ^ c), missFrac d D (c, s) ≤
      (1 - (D.card : ℝ) / 2 ^ d) ^ (2 ^ c) := by
    intro c hc
    have hcd := strata_le Q d c hc
    refine (prod_le_mean_pow (2 ^ c) (Nat.two_pow_pos c) _
      (fun s _ => missFrac_nonneg d D _)).trans (le_of_eq ?_)
    rw [sum_missFrac d c hcd D hD]
    push_cast
    rw [mul_div_cancel_left₀ _ (pow_ne_zero _ two_ne_zero)]
  calc ((strata Q d).map fun c => ∏ s ∈ Finset.range (2 ^ c), missFrac d D (c, s)).prod
      ≤ ((strata Q d).map fun c => (1 - (D.card : ℝ) / 2 ^ d) ^ (2 ^ c)).prod :=
        list_prod_map_le _ _ _ (fun c _ => Finset.prod_nonneg fun s _ => missFrac_nonneg d D _) hδ
    _ = (1 - (D.card : ℝ) / 2 ^ d) ^ ((strata Q d).map (2 ^ ·)).sum := list_prod_map_pow _ _
    _ = (1 - (D.card : ℝ) / 2 ^ d) ^ Q := by rw [strata_sum]

end FlockSoundness
